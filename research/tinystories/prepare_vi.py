"""Prepare a Vietnamese TinyStories-like corpus for AIS3 PLE training.

Input: UTF-8 plain text with a blank line between stories.
Output: data/vi_tinystories/vocab-N/{tokenizer.json,train.bin,val.bin,metadata.json}.
Train/validation are split by whole story; ids are uint16.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import unicodedata
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[2]
EOT = "<|endoftext|>"


def stories_from_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    # An empty line separates complete stories; never split a story across sets.
    docs = re.split(r"\n[ \t]*\n+", text)
    seen: set[str] = set()
    stories = []
    for doc in docs:
        doc = doc.strip()
        if not doc or doc in seen:
            continue
        if EOT in doc:
            raise SystemExit("Input contains reserved token <|endoftext|>; remove it.")
        seen.add(doc)
        stories.append(doc)
    if len(stories) < 2:
        raise SystemExit("At least two nonduplicate stories separated by a blank line are needed.")
    return stories


def make_tokenizer(train_docs: list[str], vocab: int) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[EOT],
        show_progress=True,
    )
    tok.train_from_iterator(train_docs, trainer=trainer, length=len(train_docs))
    return tok


def encode_docs(tok: Tokenizer, docs: list[str]) -> np.ndarray:
    eot_id = tok.token_to_id(EOT)
    if eot_id is None:
        raise SystemExit("Tokenizer does not contain <|endoftext|>.")
    ids: list[int] = []
    for story in docs:
        ids.extend(tok.encode(story, add_special_tokens=False).ids)
        ids.append(eot_id)
    if not ids or max(ids) > 65535:
        raise SystemExit("Invalid token ids for uint16 bins.")
    return np.asarray(ids, dtype=np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--tokenizer", type=Path, help="Reuse exactly this tokenizer for checkpoint fine-tuning")
    ap.add_argument("--output-root", type=Path, default=ROOT / "data" / "vi_tinystories")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="Explicitly replace existing prepared bins")
    args = ap.parse_args()

    if not 512 <= args.vocab <= 65536:
        raise SystemExit("--vocab must be between 512 and 65536 for uint16 tokens.")
    if not 0.0 < args.val_fraction < 0.5:
        raise SystemExit("--val-fraction must be greater than 0 and less than 0.5.")
    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    docs = stories_from_file(args.input)
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    count_val = max(1, round(len(docs) * args.val_fraction))
    train_docs, val_docs = docs[count_val:], docs[:count_val]
    out = args.output_root / f"vocab-{args.vocab}"
    targets = ["train.bin", "val.bin", "tokenizer.json", "metadata.json"]
    if not args.force and any((out / n).exists() for n in targets):
        raise SystemExit(f"{out} already has prepared files. Use --force only if replacing is intended.")
    out.mkdir(parents=True, exist_ok=True)

    if args.tokenizer:
        tok = Tokenizer.from_file(str(args.tokenizer))
        n_vocab = tok.get_vocab_size()
        if n_vocab > args.vocab:
            raise SystemExit(f"Tokenizer has {n_vocab} entries; --vocab={args.vocab} is too small.")
        if tok.token_to_id(EOT) is None:
            raise SystemExit(f"{args.tokenizer} lacks {EOT}.")
        if args.tokenizer.resolve() != (out / "tokenizer.json").resolve():
            shutil.copyfile(args.tokenizer, out / "tokenizer.json")
    else:
        tok = make_tokenizer(train_docs, args.vocab)
        tok.save(str(out / "tokenizer.json"))

    train = encode_docs(tok, train_docs)
    val = encode_docs(tok, val_docs)
    # Training script draws seq_len+1 contiguous tokens; protect against tiny examples.
    if len(train) < 514 or len(val) < 514:
        raise SystemExit(
            f"Too few tokens (train={len(train)}, val={len(val)}). "
            "Collect more stories; each partition needs >513 tokens for seq_len 512."
        )
    train.tofile(out / "train.bin")
    val.tofile(out / "val.bin")
    sha = hashlib.sha256((out / "tokenizer.json").read_bytes()).hexdigest()
    input_sha = hashlib.sha256(args.input.read_bytes()).hexdigest()
    meta = {
        "source_file": args.input.name,
        "source_sha256": input_sha,
        "tokenizer_sha256": sha,
        "vocab_capacity": args.vocab,
        "tokenizer_entries": tok.get_vocab_size(),
        "seed": args.seed,
        "train_stories": len(train_docs),
        "val_stories": len(val_docs),
        "train_tokens": len(train),
        "val_tokens": len(val),
        "normalization": "Unicode NFC",
        "split": "by story, not by token",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Prepared {out}")
    print(f"stories train/val: {len(train_docs)}/{len(val_docs)}")
    print(f"tokens train/val : {len(train):,}/{len(val):,}")
    print(f"tokenizer SHA-256: {sha}")


if __name__ == "__main__":
    main()
