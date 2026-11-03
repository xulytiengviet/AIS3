"""Experimental Vietnamese KWS CNN training on PC; exports fully INT8 TFLite.

This is a separate audio branch, NOT a PLE TinyStories model or a Voicute TCN.
For ESP32 deployment you must reproduce EXACT host MFCC features in firmware
and verify test-vector parity and TFLite Micro operator support.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, SAMPLES, discover_clips, read_pcm16_wav

FRAME_LENGTH = 400       # 25ms
FRAME_STEP = 160         # 10ms
FFT_LENGTH = 512
MEL_BINS = 40
MFCC_BINS = 13
LOW_HZ = 80.0
HIGH_HZ = 7600.0
LOG_EPSILON = 1e-6


def make_features(waveform, tf):
    """Consistent PC feature recipe; ESP32 DSP port must match this exactly.

    Per-clip energy/power normalization is deliberately avoided so features are
    reproducible. Input waveform must be 16,000 float32 samples in [-1,1].
    """
    x = tf.convert_to_tensor(waveform, dtype=tf.float32)
    stft = tf.signal.stft(x, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP,
                          fft_length=FFT_LENGTH, pad_end=False)
    power = tf.math.square(tf.abs(stft))
    weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=MEL_BINS,
        num_spectrogram_bins=FFT_LENGTH // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=LOW_HZ,
        upper_edge_hertz=HIGH_HZ,
    )
    mel = tf.tensordot(power, weights, axes=1)
    log_mel = tf.math.log(tf.maximum(mel, LOG_EPSILON))
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :MFCC_BINS]
    return tf.expand_dims(mfcc, -1)


def build_cnn(tf, n_classes: int):
    inp = tf.keras.Input(shape=(98, MFCC_BINS, 1), name="mfcc")
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(48, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                         labels: list[str]) -> dict:
    c = len(labels)
    mat = np.zeros((c, c), dtype=np.int64)
    for expected, actual in zip(y_true, y_pred):
        mat[int(expected), int(actual)] += 1
    f1s = []
    for i in range(c):
        tp = int(mat[i, i])
        fp = int(mat[:, i].sum() - tp)
        fn = int(mat[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append((2 * precision * recall / (precision + recall))
                   if precision + recall else 0.0)
    return {
        "test_accuracy": round(float(np.mean(y_true == y_pred)), 5),
        "macro_f1": round(float(np.mean(f1s)), 5),
        "per_class_f1": {label: round(float(f1), 5) for label, f1 in zip(labels, f1s)},
        "confusion_matrix_rows_true_columns_pred": mat.tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True,
                   help="dataset/<label>/<speaker>__<take>.wav")
    p.add_argument("--out", type=Path, default=Path("artifacts/kws_vi"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or args.batch < 1:
        p.error("--epochs and --batch must be positive")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("Install a CPU/GPU TensorFlow build first: pip install 'tensorflow>=2.16,<3'") from exc

    tf.keras.utils.set_random_seed(args.seed)
    clips = discover_clips(args.data, seed=f"ais3-kws-vi-{args.seed}")
    labels = sorted({c.label for c in clips})
    lookup = {label: i for i, label in enumerate(labels)}

    # In this educational baseline, compute all features in host RAM.
    # A larger corpus should switch to batched tf.data pipelines.
    features_by_split: dict[str, list[np.ndarray]] = {p: [] for p in ("train", "val", "test")}
    targets_by_split: dict[str, list[int]] = {p: [] for p in ("train", "val", "test")}
    for clip in clips:
        audio = read_pcm16_wav(clip.path)
        feature = make_features(audio, tf).numpy()
        if feature.shape != (98, MFCC_BINS, 1):
            raise AssertionError(f"Expected (98,13,1) MFCC, received {feature.shape}")
        features_by_split[clip.split].append(feature)
        targets_by_split[clip.split].append(lookup[clip.label])
    data = {part: (np.asarray(features_by_split[part], dtype=np.float32),
                   np.asarray(targets_by_split[part], dtype=np.int64))
            for part in features_by_split}
    model = build_cnn(tf, len(labels))
    model.fit(data["train"][0], data["train"][1],
              validation_data=data["val"],
              epochs=args.epochs, batch_size=args.batch, verbose=2)
    test_x, test_y = data["test"]
    pred = np.argmax(model.predict(test_x, verbose=0), axis=1)
    metrics = evaluate_predictions(test_y, pred, labels)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    train_x = data["train"][0]
    rng = np.random.default_rng(args.seed)
    indexes = rng.choice(len(train_x), size=min(300, len(train_x)), replace=False)
    def representative():
        for i in indexes:
            yield [train_x[int(i):int(i)+1].astype(np.float32)]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    payload = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=payload)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError("Unexpected TFLite I/O tensor count")
    if inputs[0]["dtype"] != np.int8 or outputs[0]["dtype"] != np.int8:
        raise RuntimeError("Export did not produce fully INT8 input/output")
    qscale, qzero = inputs[0]["quantization"]
    if qscale <= 0:
        raise RuntimeError("INT8 feature input has invalid quantization scale")
    int8_pred = []
    for feature in test_x:
        value = np.clip(np.round(feature / qscale + qzero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inputs[0]["index"], value[None])
        interpreter.invoke()
        int8_pred.append(int(np.argmax(interpreter.get_tensor(outputs[0]["index"])[0])))
    metrics["int8_test"] = evaluate_predictions(test_y, np.asarray(int8_pred), labels)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "model-int8.tflite").write_bytes(payload)
    (out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    info = {
        "status": "unverified_on_ESP32",
        "architecture": "example Conv2D CNN, not 129098-parameter paper model and not Voicute TCN",
        "parameters": model.count_params(),
        "model_sha256": hashlib.sha256(payload).hexdigest(),
        "model_bytes": len(payload),
        "feature": {
            "recipe": "TensorFlow.signal STFT->power->mel->log->MFCC",
            "sample_rate": SAMPLE_RATE, "window_samples": SAMPLES,
            "frame_length": FRAME_LENGTH, "frame_step": FRAME_STEP,
            "fft_length": FFT_LENGTH, "mel_bins": MEL_BINS, "mfcc_bins": MFCC_BINS,
            "low_hz": LOW_HZ, "high_hz": HIGH_HZ, "log_floor": LOG_EPSILON,
            "frame_count": 98, "normalize_per_clip": False,
        },
        "labels": labels,
        "speaker_split": {"train": "70%", "val": "15%", "test": "15%",
                          "method": "stable SHA256 speaker ID; no speaker overlap"},
        "split_counts": {part: len(data[part][1]) for part in data},
        "metrics_host_not_device": metrics,
        "int8_quantization": {"input": list(inputs[0]["quantization"]),
                              "output": list(outputs[0]["quantization"])},
    }
    (out / "manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {out/'model-int8.tflite'} ({len(payload):,} bytes)")
    print(f"Test accuracy host: {metrics['test_accuracy']:.4f}; int8: {metrics['int8_test']['test_accuracy']:.4f}")
    print("ESP32-S3 DSP/TFLM port and on-device performance have NOT been validated.")


if __name__ == "__main__":
    main()
