"""Offline tests: Vietnamese text safety and speaker-disjoint WAV dataset rules."""
import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from research.tinystories.vi_text import CleanOptions, load_replacements, normalize_text
from research.kws_vi.audio import discover_clips, read_pcm16_wav, speaker_split


class TextNormalizationTests(unittest.TestCase):
    def test_nfc_and_emoji_without_losing_bytes(self):
        s = "Hoa\u0300 sen 🇻🇳 👩‍💻 hóa hoá 2 =))\r\n"
        actual = normalize_text(s)
        self.assertEqual(actual, "Hoà sen 🇻🇳 👩‍💻 hóa hoá 2 =))")
        self.assertNotIn("\ufffd", actual)

    def test_tone_orthography_does_not_collapse(self):
        self.assertEqual(normalize_text("hóa và hoá, hủy và huỷ"), "hóa và hoá, hủy và huỷ")

    def test_opt_in_slang_and_boundaries(self):
        self.assertEqual(normalize_text("ko dc j r =))", CleanOptions(expand_teencode=True)),
                         "không được gì rồi (cười)")
        self.assertEqual(normalize_text("joker ko dc"), "joker ko dc")
        self.assertEqual(normalize_text("joker ko dc", CleanOptions(expand_teencode=True)),
                         "joker không được")

    def test_html_is_opt_in(self):
        self.assertEqual(normalize_text("<b>Chào</b> 😄"), "<b>Chào</b> 😄")
        self.assertEqual(normalize_text("<b>Chào</b> 😄", CleanOptions(strip_html=True)), "Chào 😄")

    def test_custom_replacements_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "words.json"
            path.write_text('{"hoá":"hóa"}', encoding="utf-8")
            rules = load_replacements(path)
            self.assertEqual(normalize_text("hoá học", CleanOptions(replacements=rules)), "hóa học")


class SpeakerSafetyTests(unittest.TestCase):
    def test_same_speaker_always_same_split(self):
        self.assertEqual(speaker_split("s01"), speaker_split("s01"))
        self.assertIn(speaker_split("s02"), {"train", "val", "test"})

    def test_cross_label_speaker_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            speakers = {}
            for i in range(500):
                s = f"speaker{i}"
                speakers.setdefault(speaker_split(s), []).append(s)
            self.assertTrue(all(len(speakers.get(p, [])) > 0 for p in ("train", "val", "test")))
            for label in ("bat_den", "tat_den", "unknown"):
                folder = root / label
                folder.mkdir()
                for spk in sorted(speakers.keys()):
                    for person in speakers[spk][:2]:
                        (folder / f"{person}__take01.wav").touch()
            clips = discover_clips(root)
            partitions = {name: {c.speaker for c in clips if c.split == name}
                          for name in ("train", "val", "test")}
            self.assertFalse(partitions["train"] & partitions["test"])
            self.assertFalse(partitions["val"] & partitions["test"])

    def test_audio_requires_16k_mono_pcm16(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "test.wav"
            with wave.open(str(wav), "wb") as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
                f.writeframes((np.zeros(1600, dtype="<i2")).tobytes())
            audio = read_pcm16_wav(wav)
            self.assertEqual(audio.shape, (16000,))
            self.assertTrue(np.all(audio == 0))


if __name__ == "__main__":
    unittest.main()
