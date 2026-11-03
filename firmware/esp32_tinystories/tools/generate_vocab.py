"""Generate an on-device ByteLevel BPE decoder and matching prompt token ids.

Important for Vietnamese: decode([single_id]) can replace a partial UTF-8 sequence
with U+FFFD, corrupting characters that straddle token boundaries. Instead we
invert the GPT-2/ByteLevel byte-to-Unicode alphabet to recover raw bytes for
each id; their concatenation is the original UTF-8 stream.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer

SKETCH = Path(__file__).resolve().parent.parent
ROOT = SKETCH.parent.parent
DEFAULT_TOK = ROOT / "artifacts" / "tinystories" / "tokenizer.json"
DEFAULT_OUT = SKETCH / "generated" / "vocab.h"


def byte_level_inverse() -> dict[str, int]:
    """Exact byte-level alphabet used by Hugging Face's ByteLevel/GPT-2 BPE."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(map(chr, cs), bs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--prompt", default="Once upon a time",
                    help="Prompt encoded with the SAME tokenizer and embedded into firmware")
    args = ap.parse_args()
    tok = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tok.get_vocab_size()
    inverse = byte_level_inverse()
    blob = bytearray()
    offsets = [0]

    for i in range(vocab_size):
        token = tok.id_to_token(i)
        if token is None:
            raise SystemExit(f"Tokenizer has no token id {i}.")
        if token == "<|endoftext|>":
            raw = b""  # special token is a story boundary, not printable content
        else:
            try:
                raw = bytes(inverse[ch] for ch in token)
            except KeyError as exc:
                raise SystemExit(
                    f"Token {i} is not a ByteLevel BPE token; unsupported character {exc}. "
                    "Use the byte-level tokenizer from AIS3."
                ) from exc
        blob.extend(raw)
        offsets.append(len(blob))

    prompt_ids = tok.encode(args.prompt, add_special_tokens=False).ids
    if not prompt_ids or len(prompt_ids) > 64:
        raise SystemExit(f"Prompt must contain 1..64 tokens; got {len(prompt_ids)}.")
    if tok.decode(prompt_ids, skip_special_tokens=False) != args.prompt:
        raise SystemExit("Prompt cannot round-trip with the selected tokenizer.")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write("// ByteLevel BPE raw bytes and prompt ids, generated from tokenizer.json.\n")
        f.write("#ifndef VOCAB_H\n#define VOCAB_H\n")
        f.write(f"#define VOCAB_N {vocab_size}\n")
        f.write(f"static const unsigned char VOCAB_BLOB[{len(blob)}] = {{\n")
        for i in range(0, len(blob), 20):
            f.write("  " + ", ".join(str(b) for b in blob[i:i+20]) + ",\n")
        f.write("};\n")
        f.write(f"static const int VOCAB_OFF[{len(offsets)}] = {{\n")
        for i in range(0, len(offsets), 20):
            f.write("  " + ", ".join(map(str, offsets[i:i+20])) + ",\n")
        f.write("};\n")
        f.write("static const int PROMPT_IDS[] = {" + ", ".join(map(str, prompt_ids)) + "};\n")
        f.write("#endif\n")
    print(f"wrote {output}: {vocab_size} tokens, {len(blob)} UTF-8 bytes")
    print(f"prompt {args.prompt!r} -> ids {prompt_ids}")


if __name__ == "__main__":
    main()
