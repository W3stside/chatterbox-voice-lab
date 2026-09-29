"""The ORIGINAL Chatterbox, smaller: what does the same model cost on the GPU
when it is stored the way it already computes?

Why: the 0.5B Chatterbox is still the voice-quality bar (Nano and Pocket lose
to it by ear), but it pinned ~6.6 GB of the 7900 XTX. Its weights are 3.2 GB
fp32, yet every matmul already runs in fp16 under autocast — and autocast keeps
an fp16 copy of each weight for the whole generate() call, so the process holds
the model twice. Almost half of S3Gen (the 124M-param S3 tokenizer) only ever
runs once, on the clone clip.

Configs — each is supra's production ChatterboxEngine (same clause chunking,
CFM steps, SDPA patch, watermark bypass, make-up gain, cached conditionals):
  prod    as deployed: fp32 weights on the GPU, fp16 autocast.
  fp16w   loads on the CPU, stores T3's and the flow's Linear/Conv weights in
          fp16, and moves only what generate() uses to the GPU. Autocast stays
          on, and it rounds an fp32 weight to exactly that fp16 value before
          every matmul anyway, so the arithmetic is unchanged. Left fp32 on
          purpose: embeddings and norms (autocast never casts them), the HiFT
          vocoder (weight-norm parametrizations; 80 MB) and the flow decoder's
          final_proj (the decoder reports its dtype from it, and that dtype is
          the ODE state's precision). The S3 tokenizer, CAMPPlus and VE stay on
          the CPU: the clone's conditionals come from the file prod saved.
  hybrid  fp16w's T3 + Nano's 2-step meanflow flow (no CFG). Its tokenizer,
          speaker encoder and vocoder are bit-identical to the original's; the
          weights are already in the HF cache.
  fp16w_nobench  fp16w with cudnn.benchmark off: MIOpen picks conv kernels by
          heuristic instead of trying (and loading) every candidate per shape —
          aimed at the VRAM that sits outside torch's allocator.

Every clause is seeded from (line, take, clause), so configs that share T3's
arithmetic must emit the SAME speech tokens — `score` checks that.

  ROCM=/home/supra/supra_project/supra_env/bin/python
  $ROCM footprint.py render --config prod --out out/footprint/RUN    # first: saves conds.pt
  $ROCM footprint.py render --config fp16w --out out/footprint/RUN
  $ROCM footprint.py server --out out/footprint/RUN                  # the live car voice
  .venv-nano/bin/python footprint.py score --out out/footprint/RUN

`server` renders through any running tts_server the way the car asks (streamed,
at the car's length scale): the live :8085 by default, or a scratch one with
--url/--name; --pid samples that server's VRAM while it speaks.
"""
import argparse
import ast
import gc
import glob
import json
import os
import sys
import threading
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUPRA_REPO = os.environ.get("SUPRA_REPO", "/home/ghost/supra-ai/supra-b58g2-ai")
sys.path[:0] = [
    os.path.join(SUPRA_REPO, "packages/voice/src"),
    os.path.join(SUPRA_REPO, "packages/contracts/src"),
]
# Everything this rig loads is already in the HF cache; never fetch.
os.environ["HF_HUB_OFFLINE"] = "1"

# The Chatterbox supra-tts unit's environment (deploy/ghost/systemd/
# supra-tts.service before ff6c590), so `prod` is measured as it ran.
_ROCM_ENV = {
    "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
    "TORCH_BLAS_PREFER_HIPBLASLT": "0",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
    "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
    "FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE",
    "FLASH_ATTENTION_TRITON_AMD_AUTOTUNE": "FALSE",
    "MIOPEN_LOG_LEVEL": "3",
}

CONFIGS = ("prod", "fp16w", "hybrid", "fp16w_nobench")
NANO_REPO = "models--ResembleAI--chatterbox-nano"
# The car asks for this pace (constants.VOICE_LENGTH_SCALE), for every engine.
CAR_LENGTH_SCALE = 0.92
LISTEN_LUFS = -18.0
POCKET_URL = "http://127.0.0.1:8085/api/tts"


def _bakeoff_lines():
    """bakeoff.LINES without importing bakeoff.py — it hides the GPU at import."""
    tree = ast.parse(open(os.path.join(HERE, "bakeoff.py")).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "LINES":
            return ast.literal_eval(node.value)
    raise RuntimeError("LINES not found in bakeoff.py")


LINES = _bakeoff_lines()


def _mib(n):
    return round(n / 2**20, 1)


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


class _VramProbe:
    """This process's VRAM as the driver counts it (KFD sysfs): the HIP context,
    kernels, BLAS/MIOpen workspaces and the torch allocator's cached blocks —
    what the other tenants of the card actually lose. Sampled every 5 ms for the
    peak, plus named snapshots with torch's own counters alongside."""

    def __init__(self):
        import torch

        self._cuda = torch.cuda
        torch.zeros(1, device="cuda")  # opens /dev/kfd, which creates the sysfs node
        paths = glob.glob(f"/sys/class/kfd/kfd/proc/{os.getpid()}/vram_*")
        if not paths:
            raise RuntimeError("no KFD sysfs VRAM node for this process")
        self._path = paths[0]
        self.peak = 0
        self.marks = {}
        self._stop = threading.Event()
        threading.Thread(target=self._poll, daemon=True).start()

    def read(self):
        with open(self._path) as f:
            return int(f.read())

    def _poll(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.read())
            self._stop.wait(0.005)

    def mark(self, name):
        self._cuda.synchronize()
        now = self.read()
        self.peak = max(self.peak, now)
        row = dict(kfd_mib=_mib(now), peak_kfd_mib=_mib(self.peak),
                   torch_alloc_mib=_mib(self._cuda.memory_allocated()),
                   torch_reserved_mib=_mib(self._cuda.memory_reserved()))
        self.marks[name] = row
        print(f"[vram] {name:9s} kfd {row['kfd_mib']:7.0f} MiB  peak {row['peak_kfd_mib']:7.0f}"
              f"  torch alloc {row['torch_alloc_mib']:7.0f} reserved {row['torch_reserved_mib']:7.0f}",
              flush=True)

    def stop(self):
        self._stop.set()


def _meanflow_s3gen():
    """Nano's S3Gen (meanflow flow, 2 steps, no CFG), loaded the way
    ChatterboxTurboTTS.from_local loads it. CPU, fp32 — _slim moves it."""
    from chatterbox.models.s3gen import S3Gen
    from safetensors.torch import load_file

    snap = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/{NANO_REPO}/snapshots/*"))[0]
    s3gen = S3Gen(meanflow=True)
    s3gen.load_state_dict(load_file(os.path.join(snap, "s3gen_meanflow.safetensors")), strict=True)
    return s3gen.eval()


def _slim(model, config):
    """Put only what generate() touches on the GPU, with fp16-stored weights
    wherever autocast would have cast them to fp16 anyway."""
    import torch
    from chatterbox.models.s3gen.s3gen import S3Token2Mel

    # S3Gen reports its device from the tokenizer's params; the tokenizer stays
    # on the CPU now, so read it from the flow instead.
    S3Token2Mel.device = property(lambda self: next(self.flow.parameters()).device)
    if config == "hybrid":
        model.s3gen = _meanflow_s3gen()
    keep_fp32 = {id(model.s3gen.flow.decoder.estimator.final_proj)}
    half_types = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.ConvTranspose1d)
    for root in (model.t3, model.s3gen.flow):
        for mod in root.modules():
            if isinstance(mod, half_types) and id(mod) not in keep_fp32:
                mod.to(torch.float16)
    model.t3.to("cuda")
    model.s3gen.flow.to("cuda")
    model.s3gen.mel2wav.to("cuda")
    model.s3gen.trim_fade = model.s3gen.trim_fade.to("cuda")
    model.device = "cuda"
    model.conds.to("cuda")
    gc.collect()
    torch.cuda.empty_cache()


def _build_engine(config, out, probe):
    """supra's ChatterboxEngine, with load/conds hooks for the config."""
    from chatterbox.tts import ChatterboxTTS, Conditionals
    from supra_voice.tts_engine import ChatterboxEngine

    conds_path = os.path.join(out, "conds.pt")
    stock_from_pretrained = ChatterboxTTS.from_pretrained.__func__
    stock_prepare = ChatterboxTTS.prepare_conditionals
    load_device = "cuda" if config == "prod" else "cpu"

    def _from_pretrained(cls, device):
        model = stock_from_pretrained(cls, load_device)
        probe.mark("loaded")
        return model

    def _prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        if config == "prod":
            stock_prepare(self, wav_fpath, exaggeration=exaggeration)
            if not os.path.exists(conds_path):
                self.conds.save(conds_path)
        else:
            # The same conditionals prod computed on the GPU, so any difference
            # downstream is the weights' storage, not the clone.
            self.conds = Conditionals.load(conds_path)
        probe.mark("conds")

    ChatterboxTTS.from_pretrained = classmethod(_from_pretrained)
    ChatterboxTTS.prepare_conditionals = _prepare_conditionals

    class _Engine(ChatterboxEngine):
        def _warmup(self):
            if config != "prod":
                _slim(self._model, config)
                probe.mark("slimmed")
            if config.endswith("_nobench"):
                import torch

                torch.backends.cudnn.benchmark = False
            super()._warmup()

    return _Engine()


def _seed_and_record(engine):
    """Seed every clause from (line, take, clause) and keep T3's speech tokens."""
    import torch

    model = engine._model
    state = dict(base=0, clause=0, tokens=[])
    stock_generate = model.generate
    stock_inference = model.t3.inference

    def _generate(text, **kwargs):
        torch.manual_seed(state["base"] + state["clause"])
        state["clause"] += 1
        return stock_generate(text, **kwargs)

    def _inference(*args, **kwargs):
        tokens = stock_inference(*args, **kwargs)
        state["tokens"].append(tokens[0].detach().cpu().tolist())
        return tokens

    model.generate = _generate
    model.t3.inference = _inference
    return state


def _render(args):
    for key, value in _ROCM_ENV.items():
        os.environ.setdefault(key, value)
    import torch

    os.makedirs(args.out, exist_ok=True)
    if args.config != "prod" and not os.path.exists(os.path.join(args.out, "conds.pt")):
        sys.exit("render --config prod first: it saves the clone's conditionals")
    probe = _VramProbe()
    probe.mark("context")
    t0 = time.perf_counter()
    engine = _build_engine(args.config, args.out, probe)
    load_s = time.perf_counter() - t0
    probe.mark("warm")
    print(f"[{args.config}] load+warmup {load_s:.1f}s", flush=True)

    state = _seed_and_record(engine)
    rows, tokens = [], {}
    for take in range(args.takes):
        for index, (key, text) in enumerate(LINES):
            state.update(base=10_000 * take + 100 * index, clause=0, tokens=[])
            chunks, first_s = [], None
            t0 = time.perf_counter()
            for pcm in engine.synthesize(text, length_scale=CAR_LENGTH_SCALE):
                if first_s is None:
                    first_s = time.perf_counter() - t0
                chunks.append(pcm)
            synth_s = time.perf_counter() - t0
            pcm = np.concatenate(chunks) if chunks else np.zeros(0, np.int16)
            audio_s = len(pcm) / engine.sample_rate
            _write_wav(os.path.join(args.out, f"{key}_{args.config}_t{take}.wav"), pcm,
                       engine.sample_rate)
            tokens[f"{key}/t{take}"] = state["tokens"]
            rows.append(dict(key=key, take=take, first_audio_s=round(first_s or 0.0, 3),
                             synth_s=round(synth_s, 3), audio_s=round(audio_s, 3),
                             rtf=round(synth_s / audio_s, 3) if audio_s else None))
            print(f"[{args.config}] t{take} {key:12s} first {first_s:5.2f}s  synth {synth_s:5.2f}s"
                  f"  audio {audio_s:5.2f}s  RTF {synth_s / max(audio_s, 1e-6):.2f}", flush=True)
    probe.mark("rendered")
    gc.collect()
    torch.cuda.empty_cache()
    probe.mark("trimmed")
    probe.stop()

    result = dict(config=args.config, sample_rate=engine.sample_rate, takes=args.takes,
                  length_scale=CAR_LENGTH_SCALE, load_s=round(load_s, 1),
                  peak_kfd_mib=_mib(probe.peak), vram=probe.marks,
                  torch_peak_alloc_mib=_mib(torch.cuda.max_memory_allocated()),
                  torch_peak_reserved_mib=_mib(torch.cuda.max_memory_reserved()),
                  lines=rows)
    with open(os.path.join(args.out, f"render_{args.config}.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.out, f"tokens_{args.config}.json"), "w") as f:
        json.dump(tokens, f)
    print(f"[{args.config}] done, peak KFD {result['peak_kfd_mib']:.0f} MiB", flush=True)


def _kfd_vram(pid):
    paths = glob.glob(f"/sys/class/kfd/kfd/proc/{pid}/vram_*")
    if not paths:
        return 0
    with open(paths[0]) as f:
        return int(f.read())


def _server(args):
    """The same lines from a running tts_server, exactly as the car asks for
    them: streamed PCM16 at the car's length scale. First audio is the first
    streamed chunk, as the Pi hears it."""
    import urllib.request

    os.makedirs(args.out, exist_ok=True)
    peak = dict(bytes=_kfd_vram(args.pid) if args.pid else 0)
    stop = threading.Event()

    def _poll():
        while not stop.is_set():
            peak["bytes"] = max(peak["bytes"], _kfd_vram(args.pid))
            stop.wait(0.005)

    if args.pid:
        threading.Thread(target=_poll, daemon=True).start()
    rows = []
    for take in range(args.takes):
        for key, text in LINES:
            body = json.dumps({"text": text, "stream": True,
                               "length_scale": CAR_LENGTH_SCALE}).encode()
            req = urllib.request.Request(args.url, data=body,
                                         headers={"Content-Type": "application/json"})
            t0 = time.perf_counter()
            first_s, parts = None, []
            with urllib.request.urlopen(req, timeout=120) as r:
                sr = int(r.headers.get("X-Sample-Rate"))
                while chunk := r.read1(65536):
                    if first_s is None:
                        first_s = time.perf_counter() - t0
                    parts.append(chunk)
            synth_s = time.perf_counter() - t0
            raw = b"".join(parts)
            pcm = np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.int16)
            _write_wav(os.path.join(args.out, f"{key}_{args.name}_t{take}.wav"), pcm, sr)
            rows.append(dict(key=key, take=take, first_audio_s=round(first_s or 0.0, 3),
                             synth_s=round(synth_s, 3), audio_s=round(len(pcm) / sr, 3)))
            print(f"[{args.name}] t{take} {key:12s} first {first_s or 0:5.2f}s  "
                  f"{len(pcm) / sr:5.2f}s audio in {synth_s:5.2f}s", flush=True)
    stop.set()
    result = dict(config=args.name, url=args.url, lines=rows)
    if args.pid:
        result["peak_kfd_mib"] = _mib(peak["bytes"])
        print(f"[{args.name}] server peak VRAM (KFD) {result['peak_kfd_mib']:.0f} MiB", flush=True)
    with open(os.path.join(args.out, f"render_{args.name}.json"), "w") as f:
        json.dump(result, f, indent=2)


def _snr_db(a, b):
    if len(a) != len(b):
        return None
    err = float(np.sum((a - b) ** 2))
    return float("inf") if err == 0.0 else round(10 * np.log10(float(np.sum(a**2)) / err), 1)


def _score(args):
    """Whisper WER, similarity to the clone clip (Chatterbox's own VE), native
    loudness, token/waveform parity vs prod, and a loudness-matched A/B."""
    import librosa
    import pyloudnorm
    import torch
    from chatterbox.models.voice_encoder import VoiceEncoder
    from faster_whisper import WhisperModel
    from safetensors.torch import load_file
    from supra_contracts import constants as c
    from supra_voice import tts_engine as te

    sys.path.insert(0, HERE)
    from bakeoff import _wer  # noqa: E402 — CPU-only scorer; hiding the GPU is fine here

    torch.set_num_threads(args.threads)
    snap = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/{NANO_REPO}/snapshots/*"))[0]
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(os.path.join(snap, "ve.safetensors")))
    ve.eval()
    ref16, _ = librosa.load(te._resolve_clone_ref(c.CHATTERBOX_CLONE_REF), sr=16000)
    ref_emb = ve.embeds_from_wavs([ref16], sample_rate=16000).mean(axis=0)
    whisper = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=args.threads)
    meters = {}

    def _measure(path, text):
        audio, sr = _read_wav_float(path)
        a16 = audio if sr == 16000 else librosa.resample(audio, orig_sr=sr, target_sr=16000)
        heard = " ".join(s.text for s in whisper.transcribe(a16, language="en", beam_size=5)[0]).strip()
        emb = ve.embeds_from_wavs([a16], sample_rate=16000).mean(axis=0)
        sim = float(np.dot(emb, ref_emb) / (np.linalg.norm(emb) * np.linalg.norm(ref_emb)))
        meter = meters.setdefault(sr, pyloudnorm.Meter(sr, block_size=0.2))
        try:
            lufs = float(meter.integrated_loudness(audio))
        except ValueError:  # shorter than one block
            lufs = float("nan")
        return dict(heard=heard, wer=round(_wer(text, heard), 3), sim=round(sim, 3),
                    lufs=round(lufs, 1), audio_s=round(len(audio) / sr, 3))

    # Every engine name with renders here: configs, and server renders by --name.
    named = {os.path.basename(p)[len(key) + 1:-len("_t0.wav")]
             for key, _text in LINES for p in glob.glob(os.path.join(args.out, f"{key}_*_t0.wav"))}
    present = [n for n in CONFIGS + ("pocket",) if n in named] + sorted(
        named - set(CONFIGS) - {"pocket"})
    tokens = {n: json.load(open(p)) for n in CONFIGS
              if os.path.exists(p := os.path.join(args.out, f"tokens_{n}.json"))}
    long_keys = {key for key, text in LINES if len(text) > 60}
    per = {n: [] for n in present}
    parity = {}
    for key, text in LINES:
        for name in present:
            for path in sorted(glob.glob(os.path.join(args.out, f"{key}_{name}_t*.wav"))):
                take = path.rsplit("_t", 1)[1][:-4]
                m = _measure(path, text)
                per[name].append(dict(key=key, take=int(take), **m))
                if name != "prod" and name in tokens and "prod" in tokens:
                    ref_path = os.path.join(args.out, f"{key}_prod_t{take}.wav")
                    same = tokens[name][f"{key}/t{take}"] == tokens["prod"][f"{key}/t{take}"]
                    snr = _snr_db(_read_wav_float(path)[0], _read_wav_float(ref_path)[0])
                    parity.setdefault(name, []).append(dict(key=key, take=int(take),
                                                            tokens_identical=same, snr_db=snr))

    summary = {}
    for name, rows in per.items():
        longs = [r for r in rows if r["key"] in long_keys]
        summary[name] = dict(
            takes=len(rows),
            sim_long=round(float(np.mean([r["sim"] for r in longs])), 3),
            sim_long_min=round(float(np.min([r["sim"] for r in longs])), 3),
            wer_all=round(float(np.mean([r["wer"] for r in rows])), 3),
            wer_long=round(float(np.mean([r["wer"] for r in longs])), 3),
            lufs=round(float(np.nanmean([r["lufs"] for r in rows])), 1),
            audio_s_per_take=round(sum(r["audio_s"] for r in rows) * len(LINES) / len(rows), 1))
    for name, rows in parity.items():
        summary[name]["tokens_identical"] = f"{sum(r['tokens_identical'] for r in rows)}/{len(rows)}"
        snrs = [r["snr_db"] for r in rows if r["tokens_identical"] and r["snr_db"] is not None]
        summary[name]["snr_db_min_when_identical"] = min(snrs) if snrs else None

    # Listening files: each line, take 0, loudness-matched, a beat between.
    # AB_all walks every engine; the pair file is the actual decision — the
    # slim Chatterbox against the live car voice.
    listen = os.path.join(args.out, "listen")
    os.makedirs(listen, exist_ok=True)
    sr_ab = 24000
    order = [n for n in ("prod", "fp16w", "fp16w_nobench", "hybrid", "pocket") if n in present] + [
        n for n in present if n not in CONFIGS + ("pocket",)]
    matched = {}
    for key, _text in LINES:
        for name in order:
            audio, sr = _read_wav_float(os.path.join(args.out, f"{key}_{name}_t0.wav"))
            if sr != sr_ab:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_ab)
            lufs = next(r["lufs"] for r in per[name] if r["key"] == key and r["take"] == 0)
            gain = 10 ** ((LISTEN_LUFS - lufs) / 20) if np.isfinite(lufs) else 1.0
            matched[key, name] = np.tanh(audio * gain)  # soft limit, never hard-clips
            _write_wav(os.path.join(listen, f"{key}_{name}.wav"),
                       (matched[key, name] * 32767).astype(np.int16), sr_ab)
    beat, gap = np.zeros(int(0.5 * sr_ab), np.float32), np.zeros(int(1.0 * sr_ab), np.float32)
    for label, names in (("all", order), ("fp16w_nobench_then_pocket", ["fp16w_nobench", "pocket"])):
        if not all(n in present for n in names):
            continue
        parts = []
        for key, _text in LINES:
            for name in names:
                parts += [matched[key, name], beat]
            parts.append(gap)
        _write_wav(os.path.join(listen, f"AB_{label}.wav"),
                   (np.concatenate(parts) * 32767).astype(np.int16), sr_ab)

    with open(os.path.join(args.out, "scores.json"), "w") as f:
        json.dump(dict(summary=summary, parity=parity, per=per), f, indent=2)
    print(json.dumps(summary, indent=2))
    print("\nheard (long lines, take 0):")
    for key, _text in LINES:
        if key in long_keys:
            for name in order:
                row = next(r for r in per[name] if r["key"] == key and r["take"] == 0)
                print(f"  {key:12s} {name:7s} {row['heard']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["render", "server", "score"])
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--takes", type=int, default=None,
                    help="takes per line (render: 3, server: 1)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--url", default=POCKET_URL, help="server: its /api/tts")
    ap.add_argument("--name", default="pocket", help="server: the name its renders get")
    ap.add_argument("--pid", type=int, default=None, help="server: its PID, to sample VRAM")
    args = ap.parse_args()
    if args.cmd == "render":
        if args.config is None:
            ap.error("render needs --config")
        args.takes = 3 if args.takes is None else args.takes
        _render(args)
    elif args.cmd == "server":
        args.takes = 1 if args.takes is None else args.takes
        _server(args)
    else:
        _score(args)


if __name__ == "__main__":
    main()
