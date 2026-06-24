# chatterbox-voice-lab

Isolated test rig for a **voice-cloned** TTS service using
[Chatterbox](https://github.com/resemble-ai/chatterbox) (MIT, zero-shot clone from a
~10s reference clip). Standalone on purpose — nothing here touches the `supra` repo.

It mirrors supra's TTS contract so a proven result ports back by copy-paste:

- **Engine** — `synthesize(text, length_scale=None)` yields 1-D int16 mono PCM chunks;
  exposes `.sample_rate` and `.default_length_scale` (same shape as supra
  `voice/tts_engine.py`).
- **HTTP** — `POST /tts {text, length_scale?}` returns Opus (`?format=wav` for parity),
  mirroring supra `web_console.py` `/api/tts`.

## Where it runs
- **Author on the MacBook** (here). No ROCm — the engine auto-detects device
  (`cuda`/hip → `mps` → `cpu`), so a slow **CPU/MPS smoke test** proves the clone +
  Opus + browser playback paths before involving the GPU.
- **Run + bench on the 7900 ROCm box.** Clone/rsync over; only the torch install differs.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# torch is host-specific (see requirements.txt header):
pip install torch torchaudio                     # Mac (CPU/MPS)
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2   # 7900
```

Drop a ~10s clean reference clip at `refs/voice.wav` (the voice you want to clone).
Legal: own/licensed voice only.

## Use
```bash
# one-shot synth + timing (first-audio latency + RTF)
python cli.py --ref refs/voice.wav --text "Oil temp climbing. Ease off for a lap."

# HTTP server + browser test page
CLONE_REF=refs/voice.wav ./run.sh           # serves :8000
#   GET  /health        -> device / sample_rate / clone_ref
#   POST /tts           -> Opus  (curl below)
#   open http://localhost:8000/ui/test.html
curl -s -XPOST localhost:8000/tts -H 'content-type: application/json' \
     -d '{"text":"Oil temp climbing."}' -o out.opus
curl -s -XPOST 'localhost:8000/tts?format=wav' -H 'content-type: application/json' \
     -d '{"text":"Oil temp climbing."}' -o out.wav      # compare sizes (~16x)
```

## What "good" looks like
- `/health` shows the **GPU** on the 7900 box (not `cpu`).
- First-audio per clause **< ~1.5 s**, **RTF < 1** on gfx1100.
- Opus ≈ 0.18 MB/min vs WAV ≈ 2.88 MB/min (~16× less) → ~1–2 MB for a 1-hr session.
- Browser/phone test page: first clause plays before the rest finish synthesizing.

## Not in scope
No changes to supra. Porting `ChatterboxEngine` into `voice/tts_engine.py` and adding
Opus streaming to `/api/tts` is the follow-up once this proves out.
