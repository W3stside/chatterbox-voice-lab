"""CLI: load reference -> clone -> synth a line -> write out.{wav,opus}; print timing.

    python cli.py --ref refs/voice.wav --text "oil temp climbing, ease off"

Reports load+warmup, first-audio latency (time to the first clause) and RTF
(synth time / audio duration; < 1.0 means faster than realtime).
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO)  # surface engine INFO (device, flow-steps, warmup)

from engine import ChatterboxEngine
from server import _encode_opus, _encode_wav  # reuse the same encoders


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ref",
        default=None,
        help="reference voice WAV (else CLONE_REF env, else Chatterbox default voice)",
    )
    ap.add_argument("--text", default="Oil temperature is climbing. Ease off for a lap.")
    ap.add_argument("--out", default="out")
    ap.add_argument(
        "--runs",
        type=int,
        default=1,
        help="synth N times (post-warmup) and report median first-audio + RTF",
    )
    args = ap.parse_args()

    t0 = time.time()
    engine = ChatterboxEngine(clone_ref=args.ref)
    print(
        f"[load+warmup] {time.time() - t0:.2f}s  device={engine.device}  sr={engine.sample_rate}"
    )

    first_audios: list[float] = []
    rtfs: list[float] = []
    pcm = np.zeros(0, dtype=np.int16)

    for run in range(max(1, args.runs)):
        t1 = time.time()
        first_audio_s = None
        chunks = []
        for ch in engine.synthesize(args.text):
            if first_audio_s is None:
                first_audio_s = time.time() - t1
            chunks.append(ch)
        run_pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        total_s = time.time() - t1
        audio_s = len(run_pcm) / engine.sample_rate if engine.sample_rate else 0.0
        rtf = (total_s / audio_s) if audio_s else float("nan")
        first_audios.append(first_audio_s or 0.0)
        rtfs.append(rtf)
        pcm = run_pcm  # keep the last run's audio to write out
        print(
            f"[run {run + 1}/{args.runs}] first-audio {first_audio_s:.2f}s  "
            f"synth {total_s:.2f}s for {audio_s:.2f}s audio  RTF={rtf:.2f}"
        )

    with open(f"{args.out}.wav", "wb") as f:
        f.write(_encode_wav(pcm, engine.sample_rate))
    try:
        with open(f"{args.out}.opus", "wb") as f:
            f.write(_encode_opus(pcm, engine.sample_rate))
        opus_note = f"{args.out}.opus"
    except Exception as e:  # noqa: BLE001
        opus_note = f"(opus skipped: {e!r})"

    if args.runs > 1:
        print(
            f"[median] first-audio {sorted(first_audios)[len(first_audios) // 2]:.2f}s  "
            f"RTF={sorted(rtfs)[len(rtfs) // 2]:.2f}  (over {args.runs} runs)"
        )
    print(f"[wrote]  {args.out}.wav  {opus_note}")
    print(f"[target] first-audio < ~1.5s and RTF < 1.0 on the 7900 → cloned voice can go live")


if __name__ == "__main__":
    main()
