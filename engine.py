"""ChatterboxEngine — mirrors the supra ``voice/tts_engine.py`` contract so a
proven result ports back into supra by copy-paste:

    synthesize(text, length_scale=None) -> yields 1-D int16 mono numpy chunks
    .sample_rate          (int, Hz)
    .default_length_scale (float)

Zero-shot voice clone: point ``CLONE_REF`` at a ~10s reference WAV.

Device auto-detect: cuda/hip -> mps -> cpu, so the same code smoke-tests on a Mac
(slow, correctness only) and runs fast on the ROCm 7900 box with no edits.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger("chatterbox_lab.engine")

# Split into clauses so audio starts before the whole reply is synthesized
# (mirrors supra's KokoroEngine clause-splitting).
_CLAUSE_SPLIT = re.compile(r"(?<=[.!?;:])\s+")

# Chatterbox has no Piper-style length_scale; kept only for contract parity.
_DEFAULT_LENGTH_SCALE = 1.0


def _pick_device() -> str:
    import torch

    # ROCm reports through the CUDA API, so this covers the 7900 box too.
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _patch_torch_load_for(device: str) -> None:
    """Chatterbox checkpoints are saved on CUDA tensors; on mps/cpu ``torch.load``
    needs an explicit ``map_location`` or it raises. Patch once, idempotently."""
    if device == "cuda":
        return
    import torch

    if getattr(torch, "_chatterbox_lab_patched", False):
        return
    _orig = torch.load
    map_loc = torch.device(device)

    def _patched(*args, **kwargs):
        kwargs.setdefault("map_location", map_loc)
        return _orig(*args, **kwargs)

    torch.load = _patched
    torch._chatterbox_lab_patched = True


def _split_clauses(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(text) if p.strip()]
    return parts or [text]


# Keep the FIRST audible chunk short so first-audio latency is low; later chunks stay
# clause-sized (they synthesize while the first one plays). Set =0 to disable.
_FIRST_CHUNK_CHARS = int(os.environ.get("CHATTERBOX_FIRST_CHUNK_CHARS", "48"))


def _split_at(s: str, limit: int) -> tuple[str, str]:
    """Split ``s`` near ``limit`` chars on the last comma/space boundary <= limit."""
    if len(s) <= limit:
        return s, ""
    window = s[:limit]
    cut = max(window.rfind(", "), window.rfind(" "))
    if cut <= 0:
        cut = limit
    return s[:cut].strip(" ,"), s[cut:].strip(" ,")


def _chunks(text: str) -> list[str]:
    """Clause split, but shorten the first chunk for fast first-audio."""
    clauses = _split_clauses(text)
    if not clauses:
        return []
    first, rest = clauses[0], clauses[1:]
    if _FIRST_CHUNK_CHARS > 0 and len(first) > _FIRST_CHUNK_CHARS:
        head, tail = _split_at(first, _FIRST_CHUNK_CHARS)
        return [c for c in (head, tail) if c] + rest
    return clauses


class ChatterboxEngine:
    """Zero-shot cloning TTS engine. One instance per process; load once."""

    def __init__(
        self,
        clone_ref: Optional[str] = None,
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        from chatterbox.tts import ChatterboxTTS

        import torch

        self.device = device or _pick_device()
        _patch_torch_load_for(self.device)

        # ROCm/gfx1100 (RDNA3) perf flags — only on a real HIP GPU. The Mac CPU/MPS
        # venv has torch.version.hip is None and must stay untouched.
        self._is_rocm = self.device == "cuda" and getattr(torch.version, "hip", None) is not None
        if self._is_rocm:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True  # autotune conv algos for bounded per-clause shapes
            torch.set_float32_matmul_precision("high")
        self._fp16 = self._is_rocm  # prefer fp16 over bf16 on RDNA3 (cleared the MIOpen workspace-0 hang)

        logger.info("loading Chatterbox on device=%s (rocm=%s)", self.device, self._is_rocm)
        self._model = ChatterboxTTS.from_pretrained(device=self.device)

        self.sample_rate: int = int(self._model.sr)
        self.default_length_scale: float = _DEFAULT_LENGTH_SCALE
        # Chatterbox generate() knobs — authoritative defaults from src/chatterbox/tts.py
        # (exaggeration=0.5, cfg_weight=0.5, temperature=0.8). Env-overridable so voice
        # character can be tuned without code edits. n_timesteps is NOT here — it's
        # hardcoded inside the s3gen flow-matching and only changeable via a source patch.
        self.exaggeration = exaggeration if exaggeration is not None else float(os.environ.get("CHATTERBOX_EXAGGERATION", "0.5"))
        self.cfg_weight = cfg_weight if cfg_weight is not None else float(os.environ.get("CHATTERBOX_CFG_WEIGHT", "0.5"))
        self.temperature = temperature if temperature is not None else float(os.environ.get("CHATTERBOX_TEMPERATURE", "0.8"))

        ref = clone_ref if clone_ref is not None else os.environ.get("CLONE_REF")
        if ref is not None and not os.path.exists(ref):
            logger.warning("CLONE_REF %r not found — using Chatterbox default voice", ref)
            ref = None
        self.clone_ref = ref
        if self.clone_ref is None:
            logger.warning("no reference clip — speaking in the default voice, not a clone")

        # Cache the reference conditioning ONCE. Chatterbox otherwise re-extracts it
        # (librosa load + conv embed + tokenizer) on EVERY generate() call when
        # audio_prompt_path is passed — pure per-clause waste. Prepare once, then
        # reuse self._model.conds (drop audio_prompt_path in _generate_one).
        self._conds_ready = False
        if self.clone_ref is not None:
            try:
                self._model.prepare_conditionals(self.clone_ref, exaggeration=self.exaggeration)
                self._conds_ready = True
            except Exception as e:  # noqa: BLE001
                logger.warning("prepare_conditionals failed; per-clause ref fallback: %r", e)

        # Unlock SDPA on the attention-heavy T3 loop. Full effect also needs the
        # t3.py output_attentions=False source patch; this set is harmless on its own.
        if self._is_rocm:
            try:
                self._model.t3.cfg._attn_implementation = "sdpa"
            except Exception as e:  # noqa: BLE001
                logger.warning("sdpa attn_implementation set failed: %r", e)

        # NOTE: S3Gen flow-matching steps (n_timesteps, default 10) are the next speed
        # lever but they're a source-level value in chatterbox flow.py, not a settable
        # attribute — reducing them needs a vendored flow.py patch. Left at stock 10;
        # warm RTF ~0.57-0.84 / first-audio ~1.0s already clears target on the 7900.

        self._warmup()

    def _warmup(self) -> None:
        """Absorb cold start (model load, fp16 paths, MIOpen/cudnn conv autotune).
        Warm a few clause lengths so cudnn.benchmark settles the bounded per-clause
        shapes before the first real synth."""
        try:
            t0 = time.time()
            for line in (
                "Warm up.",
                "Systems nominal and ready to go.",
                "Oil pressure is holding steady at four point two bar.",
            ):
                for _ in self._generate_one(line):
                    pass
            logger.info("warmup done in %.2fs", time.time() - t0)
        except Exception as e:  # noqa: BLE001
            logger.warning("warmup failed: %r", e)

    def _generate_one(self, clause: str) -> Iterator[np.ndarray]:
        import torch

        kwargs = dict(
            exaggeration=self.exaggeration,
            cfg_weight=self.cfg_weight,
            temperature=self.temperature,
        )
        # Only pass the ref path if conds caching failed; otherwise reuse cached conds.
        if self.clone_ref is not None and not getattr(self, "_conds_ready", False):
            kwargs["audio_prompt_path"] = self.clone_ref

        if getattr(self, "_fp16", False):
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    wav = self._model.generate(clause, **kwargs)
            except RuntimeError as e:  # fp16 op-unsupported -> safe fp32 retry
                logger.warning("fp16 autocast generate failed, retrying fp32: %r", e)
                wav = self._model.generate(clause, **kwargs)
        else:
            wav = self._model.generate(clause, **kwargs)

        # generate() returns a CPU fp32 (1, N) tensor (watermarker runs on numpy);
        # the int16 mono-chunk contract is unchanged.
        arr = wav.detach().to("cpu").float().numpy().reshape(-1)
        arr = np.clip(arr, -1.0, 1.0)
        yield (arr * 32767.0).astype(np.int16)

    def synthesize(
        self, text: str, length_scale: Optional[float] = None
    ) -> Iterator[np.ndarray]:
        """Yield 1-D int16 mono PCM chunks, one per clause.

        ``length_scale`` is accepted for contract parity with supra's Piper/Kokoro
        engines but has no Chatterbox analogue, so it is ignored.
        """
        for clause in _chunks(text):
            yield from self._generate_one(clause)
