"""Dataset / PCM utilities for a standalone Vietnamese keyword-spotting experiment.

Audio: WAV PCM16 mono, 16 kHz. File layout: dataset/<label>/<speaker>__<take>.wav.
Groups are split by SPEAKER, not by recording, to prevent speaker leakage.
"""
from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.0
SAMPLES = 16000


@dataclass(frozen=True)
class Clip:
    path: Path
    label: str
    speaker: str
    split: str


def speaker_split(speaker: str, seed: str = "ais3-kws-vi-v1") -> str:
    """Stable 70/15/15 train/validation/test split by speaker identity."""
    v = int.from_bytes(hashlib.sha256((seed + ":" + speaker).encode("utf-8")).digest()[:8], "big") % 100
    return "train" if v < 70 else ("val" if v < 85 else "test")


def discover_clips(dataset: Path, seed: str = "ais3-kws-vi-v1") -> list[Clip]:
    if not dataset.is_dir():
        raise ValueError(f"Dataset directory not found: {dataset}")
    clips = []
    for folder in sorted(p for p in dataset.iterdir() if p.is_dir()):
        for wav in sorted(folder.glob("*.wav")):
            if "__" not in wav.stem:
                raise ValueError(
                    f"{wav}: name must be speakerID__takeID.wav; group by actual speaker, not filename"
                )
            speaker = wav.stem.split("__", 1)[0]
            if not speaker:
                raise ValueError(f"{wav}: speaker ID cannot be empty")
            clips.append(Clip(wav, folder.name, speaker, speaker_split(speaker, seed)))
    if not clips:
        raise ValueError("No WAV clips found: expected dataset/<label>/<speaker>__<take>.wav")
    names = {c.label for c in clips}
    if len(names) < 2:
        raise ValueError("At least two labels are needed (commands plus unknown/noise)")
    for label in sorted(names):
        for partition in ("train", "val", "test"):
            if not any(c.label == label and c.split == partition for c in clips):
                raise ValueError(
                    f"Class {label!r} missing {partition}. Add more independent speakers; "
                    "do not split clips by file to mask leakage."
                )
    speakers_by_split = {part: {c.speaker for c in clips if c.split == part}
                         for part in ("train", "val", "test")}
    if any(speakers_by_split[a] & speakers_by_split[b]
           for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise AssertionError("Speaker-leakage detected")
    return clips


def read_pcm16_wav(path: Path, window_samples: int = SAMPLES) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        rate = wav.getframerate()
        width = wav.getsampwidth()
        compression = wav.getcomptype()
        if (channels, rate, width, compression) != (1, SAMPLE_RATE, 2, "NONE"):
            raise ValueError(
                f"{path} must be 16 kHz, mono, 16-bit PCM WAV; "
                f"received channels={channels}, rate={rate}, width={width}, comp={compression}"
            )
        raw = wav.readframes(wav.getnframes())
    signal = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if signal.size >= window_samples:
        # Deterministic center crop. Data augmentation must be handled separately.
        start = (signal.size - window_samples) // 2
        return signal[start:start + window_samples]
    out = np.zeros(window_samples, dtype=np.float32)
    start = (window_samples - signal.size) // 2
    out[start:start + signal.size] = signal
    return out
