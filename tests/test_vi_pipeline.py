"""Regression tests for Vietnamese UTF-8 BPE and corpus preparation.

They do not require a microcontroller, model checkpoint, or network.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research.tinystories import prepare_vi
from firmware.esp32_tinystories.tools.generate_vocab import byte_level_inverse


class VietnameseCorpusTests(unittest.TestCase):
    def test_unicode_nfc_and_deduplicate(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "input.txt"
            file.write_text(
                "Nga\u0300y xưa cậu bé đến trường.\n\n"
                "Ngày xưa cậu bé đến trường.\n\n"
                "Cô bé ngắm những bông hoa.\n",
                encoding="utf-8",
            )
            docs = prepare_vi.stories_from_file(file)
            self.assertEqual(len(docs), 2)
            self.assertEqual(docs[0], "Ngày xưa cậu bé đến trường.")

    def test_utf8_bytes_roundtrip_even_across_bpe_token_boundaries(self):
        texts = ["Ngày xửa ngày xưa, cô bé Mây ăn quả dứa."] * 32
        tok = prepare_vi.make_tokenizer(texts, vocab=512)
        inverse = byte_level_inverse()
        prompt = "Tiếng Việt có dấu: ắ, ệ, ờ, ỹ, đ."
        ids = tok.encode(prompt, add_special_tokens=False).ids
        raw = bytearray()
        for token_id in ids:
            symbol = tok.id_to_token(token_id)
            self.assertIsNotNone(symbol)
            raw.extend(inverse[ch] for ch in symbol)
        self.assertEqual(bytes(raw), prompt.encode("utf-8"))
        self.assertNotIn(b"\xef\xbf\xbd", bytes(raw), "UTF-8 replacement character appeared")

    def test_prepare_splits_by_story_and_writes_uint16(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "stories.txt"
            # A genuine training corpus must be larger; repeats here exercise file IO only.
            stories = [
                f"Truyện số {i}. " + "Ngày xưa, cô bé An học cách trồng cây. " * 120
                for i in range(16)
            ]
            src.write_text("\n\n".join(stories), encoding="utf-8")
            dst = Path(d) / "prepared"
            with patch.object(sys, "argv", [
                "prepare_vi", "--input", str(src), "--vocab", "512",
                "--output-root", str(dst),
            ]):
                prepare_vi.main()
            data = dst / "vocab-512"
            m = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(m["train_stories"] + m["val_stories"], len(stories))
            self.assertGreater(len(np.fromfile(data / "train.bin", dtype=np.uint16)), 514)
            self.assertGreater(len(np.fromfile(data / "val.bin", dtype=np.uint16)), 514)
            self.assertEqual(len(m["tokenizer_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
