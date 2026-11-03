"""Audit Vietnamese corpus and tokenizer without training or using an ESP32.

From repo root:
 uv run python -m research.tinystories.audit_vi --input data/vi_stories.txt \
   --tokenizer data/vi_tinystories/vocab-32768/tokenizer.json
Exit nonzero if tokenizer round-trip, Unicode NFC, or unknown-token checks fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import unicodedata
from pathlib import Path

from tokenizers import Tokenizer

from .prepare_vi import stories_from_file
from .vi_text import CleanOptions, load_replacements


def audit(corpus: Path, tokenizer: Path, options: CleanOptions = CleanOptions(),
          max_stories: int | None = None) -> dict:
    tok = Tokenizer.from_file(str(tokenizer))
    docs = stories_from_file(corpus, options)
    if max_stories is not None:
        docs = docs[:max_stories]
    if not docs:
        raise ValueError("No stories to audit")
    total_ids = 0
    token_per_story = []
    unknown_ids = 0
    lossless_failures: list[dict] = []
    non_nfc_stories = 0
    tokens = tok.get_vocab_size()
    unk_id = tok.token_to_id("<unk>")
    for i, story in enumerate(docs):
        non_nfc_stories += unicodedata.normalize("NFC", story) != story
        enc = tok.encode(story, add_special_tokens=False)
        token_per_story.append(len(enc.ids))
        total_ids += len(enc.ids)
        if unk_id is not None:
            unknown_ids += enc.ids.count(unk_id)
        decoded = tok.decode(enc.ids, skip_special_tokens=False)
        if decoded != story and len(lossless_failures) < 5:
            lossless_failures.append({
                "story_index": i,
                "expected_prefix": story[:120],
                "actual_prefix": decoded[:120],
            })
    chars = sum(len(s) for s in docs)
    utf8_bytes = sum(len(s.encode("utf-8")) for s in docs)
    syllables = sum(len(s.split()) for s in docs)
    report = {
        "source": corpus.name,
        "source_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
        "story_count": len(docs),
        "vocab_entries": tokens,
        "total_tokens": total_ids,
        "utf8_bytes_per_token": round(utf8_bytes / max(1, total_ids), 4),
        "tokens_per_100_space_separated_syllables_approx": round(total_ids * 100 / max(1, syllables), 2),
        "tokens_per_story_mean": round(statistics.mean(token_per_story), 2),
        "non_nfc_story_count": non_nfc_stories,
        "unknown_tokens": unknown_ids,
        "unknown_rate": round(unknown_ids / max(1, total_ids), 8),
        "round_trip_failures": len(lossless_failures),
        "round_trip_failure_samples": lossless_failures,
        "character_count": chars,
        "cleaning": {
            "lowercase": options.lowercase,
            "strip_html": options.strip_html,
            "expand_teencode": options.expand_teencode,
            "has_explicit_replacements": options.replacements is not None,
            "tone_style": "not rewritten by default",
        },
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--max-stories", type=int)
    p.add_argument("--lowercase", action="store_true")
    p.add_argument("--strip-html", action="store_true")
    p.add_argument("--expand-teencode", action="store_true")
    p.add_argument("--replacements", type=Path)
    p.add_argument("--output", type=Path, help="Optional JSON report path")
    args = p.parse_args()
    if args.max_stories is not None and args.max_stories < 1:
        p.error("--max-stories must be >= 1")
    opts = CleanOptions(
        lowercase=args.lowercase,
        strip_html=args.strip_html,
        expand_teencode=args.expand_teencode,
        replacements=load_replacements(args.replacements) if args.replacements else None,
    )
    summary = audit(args.input, args.tokenizer, opts, args.max_stories)
    output = json.dumps(summary, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    if summary["round_trip_failures"] or summary["unknown_tokens"] or summary["non_nfc_story_count"]:
        raise SystemExit("AUDIT FAILED: non-lossless tokenization, unknown tokens, or non-NFC texts")
    print("AUDIT PASS: NFC, full round-trip and no unknown token ids")


if __name__ == "__main__":
    main()
