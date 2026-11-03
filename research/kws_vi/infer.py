"""Run a calibrated INT8 Vietnamese KWS TFLite clip classification on a PC.

This is NOT a live microphone or ESP32 firmware test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audio import read_pcm16_wav
from .train import make_features


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=Path("artifacts/kws_vi"))
    p.add_argument("--wav", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.8)
    args = p.parse_args()
    if not 0 <= args.threshold <= 1:
        p.error("--threshold must be in [0,1]")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("Install TensorFlow to run the MFCC/TFLite host benchmark") from exc
    labels = json.loads((args.model_dir / "labels.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.model_dir / "manifest.json").read_text(encoding="utf-8"))
    audio = read_pcm16_wav(args.wav)
    feature = make_features(audio, tf).numpy().astype(np.float32)[None]
    interpreter = tf.lite.Interpreter(model_path=str(args.model_dir / "model-int8.tflite"))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    sc, zero = inp["quantization"]
    out_sc, out_zero = out["quantization"]
    if sc <= 0 or out_sc <= 0 or inp["dtype"] != np.int8 or out["dtype"] != np.int8:
        raise SystemExit("Expected fully quantized INT8 model with valid quantization scales")
    q = np.clip(np.rint(feature / sc + zero), -128, 127).astype(np.int8)
    if tuple(q.shape) != tuple(inp["shape"]):
        raise SystemExit(f"Feature shape mismatch: {q.shape} != {inp['shape']}")
    interpreter.set_tensor(inp["index"], q)
    interpreter.invoke()
    probs = (interpreter.get_tensor(out["index"])[0].astype(np.float32) - out_zero) * out_sc
    top = int(np.argmax(probs))
    label = labels[top] if float(probs[top]) >= args.threshold else "_reject"
    print(json.dumps({
        "detected": label, "candidate": labels[top], "confidence": round(float(probs[top]), 5),
        "threshold": args.threshold, "labels": labels,
        "model_sha256": manifest["model_sha256"],
        "runtime": "host TFLite interpreter, not ESP32-S3",
        "warning": "A single-clip score is not always-on recall or false-accepts/hour",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
