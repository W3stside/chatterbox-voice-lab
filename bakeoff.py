"""Chatterbox-Nano vs the current Chatterbox — a CPU-only bake-off.

Why: Chatterbox pins ~6.6 GB of the 7900 XTX, which is what keeps a second LLM
from fitting beside the reason model. Nano (110M T3 + the shared meanflow
decoder) claims 3x realtime on 8 CPU threads, i.e. zero VRAM. This measures
whether it keeps Katana's voice and the live-speech latency bar.

Both engines render the SAME lines on CPU (torch +cpu wheel — the GPU is never
touched, so it can't collide with whatever Ollama is doing):
  * orig — supra's production ChatterboxEngine class itself (original 0.5B
           model, CFM steps / watermark bypass / clause chunking as deployed)
  * nano — ChatterboxTurboTTS(nano=True), wrapped to match that engine
  * pocket / pocket_q — Kyutai Pocket TTS (fp32 / int8-quantized), streaming.
           Needs numpy>=2 vs chatterbox's <2, so it renders from .venv-pocket;
           `score` always runs in .venv-nano (Whisper + Chatterbox's VE)
Make-up gain is zeroed for both (it was tuned for orig's level; Nano normalises
its reference to -27 LUFS, so the two land at different levels) — `score`
reports each engine's native loudness and writes loudness-matched copies for
the listening A/B instead.

Each engine renders in its own process so its peak RSS is its own:
  .venv-nano/bin/python bakeoff.py render --engine orig --out out/bakeoff/RUN
  .venv-nano/bin/python bakeoff.py render --engine nano --out out/bakeoff/RUN
  .venv-nano/bin/python bakeoff.py score --out out/bakeoff/RUN
"""
import argparse
import json
import os
import re
import resource
import sys
import time
import wave

# Thread caps + CPU-only must be in place BEFORE torch is imported.
_THREADS = "8"
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
# Gain is applied at listening time, loudness-matched (see module docstring).
os.environ["SUPRA_CHATTERBOX_GAIN_DB"] = "0"

import numpy as np

SUPRA_REPO = os.environ.get("SUPRA_REPO", "/home/ghost/supra-ai/supra-b58g2-ai")
sys.path[:0] = [
    os.path.join(SUPRA_REPO, "packages/voice/src"),
    os.path.join(SUPRA_REPO, "packages/contracts/src"),
]

NANO_REPO = "ResembleAI/chatterbox-nano"
NANO_FILES = [
    "t3_nano_v1.safetensors", "s3gen_meanflow.safetensors", "ve.safetensors",
    "*.json", "*.txt", "*.pt", "*.model",
]

# Short cues have baked GPU renders in assets/wake/ack (stem → a third column
# in the scores); the long lines are typical Katana replies. Numbers are
# spelled out the way the brain speaks them.
LINES = [
    ("ack_1", "Comms open."),
    ("ack_3", "Listening."),
    ("ack_4", "Go ahead, sir."),
    ("followup_2", "Anything else, sir?"),
    ("again_1", "Didn't catch that. Say again?"),
    ("reason_1", "Analysing that now."),
    ("reason_3", "Running the numbers, sir."),
    ("oil", "Oil's at ninety eight degrees and coolant is ninety one, so the "
            "engine is fully warm. You're clear to push."),
    ("tyres", "Front left is sitting at thirty one PSI, about two under the "
              "others. Worth a look at the next stop."),
    ("boost", "Boost peaked at one point four bar on that last pull, and "
              "intake temps stayed under forty degrees."),
    ("reason_long", "Looking at the last three laps, your oil temperature "
                    "climbs about four degrees per lap and never recovers on "
                    "the straights. That points to heat soak rather than a "
                    "cooling fault, so a cool-down lap every fifth lap should "
                    "keep you under the limit."),
    ("banter", "Honestly? That was your cleanest run today. Keep the inputs "
               "that smooth and the tyres will thank you."),
]

LISTEN_LUFS = -18.0

# Render order = listening order in the A/B file.
ENGINES = ("orig", "nano", "pocket", "pocket_q")


def _peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _write_wav(path, pcm, sr):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.asarray(pcm, dtype=np.int16).tobytes())


def _read_wav_float(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32767.0, sr


class _NanoEngine:
    """ChatterboxEngine's shape (synthesize() → int16 clause chunks) around
    Nano. Deliberately thin: same clause chunking, same speed-bias stretch,
    same cached conditionals + watermark bypass, so any difference measured
    is the model, not the plumbing."""

    def __init__(self):
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        from huggingface_hub import snapshot_download
        from supra_contracts import constants as c
        from supra_voice import tts_engine as te

        self._te = te
        ckpt = snapshot_download(NANO_REPO, allow_patterns=NANO_FILES, local_files_only=True)
        self._model = ChatterboxTurboTTS.from_local(ckpt, "cpu", nano=True)
        self.sample_rate = int(self._model.sr)
        self.speed_bias = float(c.CHATTERBOX_SPEED_BIAS)
        self.temperature = c.CHATTERBOX_TEMPERATURE
        self._model.prepare_conditionals(te._resolve_clone_ref(c.CHATTERBOX_CLONE_REF))
        self._model.watermarker.apply_watermark = lambda wav, sample_rate=None: wav
        for line in ("Warm up.", "Systems nominal and ready to go.",
                     "Oil pressure is holding steady at four point two bar."):
            for _ in self.synthesize(line):
                pass

    def synthesize(self, text, length_scale=None):
        rate = self.speed_bias / max(1.0 if length_scale is None else float(length_scale), 0.1)
        for clause in self._te._chunks(text):
            wav = self._model.generate(clause, temperature=self.temperature)
            arr = wav.detach().to("cpu").float().numpy().reshape(-1)
            arr = np.clip(self._te._time_stretch(arr, rate), -1.0, 1.0)
            yield (arr * 32767.0).astype(np.int16)


class _PocketEngine:
    """Kyutai Pocket TTS, cloned from the same reference clip. It streams audio
    frames while the line is still being generated, so it takes the WHOLE line
    (its own chunker keeps prosody across clauses) rather than supra's clause
    split — that streaming is the first-audio win being measured. A
    phase-vocoder stretch can't run on ~80 ms frames without seams, so the
    speed bias is applied to the finished line afterwards (`stretch_after`)
    to keep the A/B at the same pace as orig/nano."""

    def __init__(self, quantize=False):
        from pocket_tts import TTSModel
        from supra_contracts import constants as c
        from supra_voice import tts_engine as te

        self._model = TTSModel.load_model(quantize=quantize)
        if not self._model.has_voice_cloning:
            # The gated cloning weights failed to download → a stock catalog
            # voice, which would make the whole comparison meaningless.
            raise RuntimeError("pocket-tts loaded WITHOUT voice cloning (HF gate/token?)")
        self.sample_rate = int(self._model.sample_rate)
        self.stretch_after = float(c.CHATTERBOX_SPEED_BIAS)
        self._voice = self._model.get_state_for_audio_prompt(
            te._resolve_clone_ref(c.CHATTERBOX_CLONE_REF), truncate=True)
        for line in ("Warm up.", "Systems nominal and ready to go.",
                     "Oil pressure is holding steady at four point two bar."):
            for _ in self.synthesize(line):
                pass

    def synthesize(self, text, length_scale=None):
        for chunk in self._model.generate_audio_stream(self._voice, text):
            arr = np.clip(chunk.detach().to("cpu").float().numpy().reshape(-1), -1.0, 1.0)
            yield (arr * 32767.0).astype(np.int16)


def _load(engine):
    if engine == "nano":
        return _NanoEngine()
    if engine in ("pocket", "pocket_q"):
        return _PocketEngine(quantize=engine == "pocket_q")
    from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
    from supra_voice.tts_engine import ChatterboxEngine

    # Production's SDPA patch reads alignment_stream_analyzer, which chatterbox
    # master dropped along with the output_attentions=True it worked around
    # (master passes False itself) — so the patch is moot here; skip it.
    T3HuggingfaceBackend._supra_sdpa_patched = True
    return ChatterboxEngine()


def _render(args):
    import torch

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    eng = _load(args.engine)
    load_s = time.perf_counter() - t0
    rss_after_load = _peak_rss_mb()
    print(f"[{args.engine}] load+warmup {load_s:.1f}s, peak RSS {rss_after_load:.0f} MB", flush=True)

    rows = []
    for key, text in LINES:
        chunks, first_s = [], None
        t0 = time.perf_counter()
        for pcm in eng.synthesize(text):
            if first_s is None:
                first_s = time.perf_counter() - t0
            chunks.append(pcm)
        synth_s = time.perf_counter() - t0
        pcm = np.concatenate(chunks) if chunks else np.zeros(0, np.int16)
        native_s = len(pcm) / eng.sample_rate
        stretch = getattr(eng, "stretch_after", None)
        if stretch is not None:
            from supra_voice import tts_engine as te

            pcm = (np.clip(te._time_stretch(pcm.astype(np.float32) / 32767.0, stretch),
                           -1.0, 1.0) * 32767.0).astype(np.int16)
        audio_s = len(pcm) / eng.sample_rate
        _write_wav(os.path.join(args.out, f"{key}_{args.engine}.wav"), pcm, eng.sample_rate)
        rows.append(dict(key=key, text=text, first_audio_s=round(first_s or 0.0, 3),
                         synth_s=round(synth_s, 3), audio_s=round(audio_s, 3),
                         native_audio_s=round(native_s, 3),
                         rtf=round(synth_s / audio_s, 3) if audio_s else None))
        print(f"[{args.engine}] {key:12s} first {first_s:5.2f}s  synth {synth_s:5.2f}s  "
              f"audio {audio_s:5.2f}s  RTF {synth_s / max(audio_s, 1e-6):.2f}", flush=True)

    result = dict(engine=args.engine, threads=args.threads, sample_rate=eng.sample_rate,
                  load_s=round(load_s, 1), rss_after_load_mb=round(rss_after_load),
                  peak_rss_mb=round(_peak_rss_mb()), lines=rows)
    with open(os.path.join(args.out, f"render_{args.engine}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.engine}] done, peak RSS {result['peak_rss_mb']} MB", flush=True)


def _words(text):
    return re.sub(r"[^a-z' ]+", " ", text.lower().replace("-", " ")).split()


def _wer(ref, hyp):
    """Word-level Levenshtein / reference length."""
    r, h = _words(ref), _words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(len(r), 1)


def _score(args):
    import librosa
    import pyloudnorm
    import torch
    from chatterbox.models.voice_encoder import VoiceEncoder
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from supra_contracts import constants as c
    from supra_voice import tts_engine as te

    torch.set_num_threads(args.threads)
    ve = VoiceEncoder()
    ckpt = snapshot_download(NANO_REPO, allow_patterns=NANO_FILES, local_files_only=True)
    ve.load_state_dict(load_file(os.path.join(ckpt, "ve.safetensors")))
    ve.eval()
    ref16, _ = librosa.load(te._resolve_clone_ref(c.CHATTERBOX_CLONE_REF), sr=16000)
    ref_emb = ve.embeds_from_wavs([ref16], sample_rate=16000).mean(axis=0)

    whisper = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=args.threads)
    meter_cache = {}

    def _measure(path):
        audio, sr = _read_wav_float(path)
        text = " ".join(s.text for s in whisper.transcribe(audio if sr == 16000 else
                        librosa.resample(audio, orig_sr=sr, target_sr=16000),
                        language="en", beam_size=5)[0]).strip()
        a16 = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        emb = ve.embeds_from_wavs([a16], sample_rate=16000).mean(axis=0)
        sim = float(np.dot(emb, ref_emb) / (np.linalg.norm(emb) * np.linalg.norm(ref_emb)))
        meter = meter_cache.setdefault(sr, pyloudnorm.Meter(sr, block_size=0.2))
        try:
            lufs = float(meter.integrated_loudness(audio))
        except ValueError:  # shorter than one block
            lufs = float("nan")
        return dict(heard=text, sim=round(sim, 3), lufs=round(lufs, 1), audio=audio, sr=sr)

    listen_dir = os.path.join(args.out, "listen")
    os.makedirs(listen_dir, exist_ok=True)
    scores, ab = [], []
    for key, text in LINES:
        row = dict(key=key)
        takes = [(e, os.path.join(args.out, f"{key}_{e}.wav")) for e in ENGINES]
        takes.append(("gpu_baked", os.path.join(SUPRA_REPO, "assets/wake/ack", f"{key}.wav")))
        for name, path in takes:
            if not os.path.exists(path):
                continue
            m = _measure(path)
            row[name] = dict(wer=round(_wer(text, m["heard"]), 2), sim=m["sim"],
                             lufs=m["lufs"], heard=m["heard"])
            if name in ENGINES:
                gain = 10 ** ((LISTEN_LUFS - m["lufs"]) / 20) if np.isfinite(m["lufs"]) else 1.0
                matched = np.tanh(m["audio"] * gain)  # soft limit, never hard-clips
                _write_wav(os.path.join(listen_dir, f"{key}_{name}.wav"),
                           (matched * 32767).astype(np.int16), m["sr"])
                ab.append((key, name, matched, m["sr"]))
        scores.append(row)

    # One file to listen through: each line in ENGINES order, a beat between.
    present = [e for e in ENGINES if any(name == e for _k, name, _a, _s in ab)]
    sr = ab[0][3]
    gap = np.zeros(int(0.5 * sr), np.float32)
    parts = []
    for _key, _name, audio, _sr in ab:
        parts += [audio, gap]
    _write_wav(os.path.join(listen_dir, f"AB_{'_'.join(present)}.wav"),
               (np.concatenate(parts) * 32767).astype(np.int16), sr)

    with open(os.path.join(args.out, "scores.json"), "w") as f:
        json.dump(scores, f, indent=2)

    cols = present + ["gpu_baked"]
    for field, fmt in (("sim", "{:6.3f}"), ("wer", "{:6.2f}"), ("lufs", "{:6.1f}")):
        print(f"\n{field:12s} " + " ".join(f"{c:>9s}" for c in cols))
        for row in scores:
            print(f"{row['key']:12s} " + " ".join(
                f"{fmt.format(row[c][field]):>9s}" if c in row else f"{'-':>9s}" for c in cols))
    print("\nheard (long lines):")
    for row in scores[7:]:
        for c in present:
            print(f"  {row['key']:12s} {c:8s} {row[c]['heard']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["render", "score"])
    ap.add_argument("--engine", choices=list(ENGINES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=int(_THREADS))
    args = ap.parse_args()
    if args.cmd == "render":
        if args.engine is None:
            ap.error("render needs --engine")
        _render(args)
    else:
        _score(args)


if __name__ == "__main__":
    main()
