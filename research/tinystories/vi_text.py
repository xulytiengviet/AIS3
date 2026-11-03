"""Conservative, configurable Vietnamese text normalization for AIS3.

Default behavior intentionally does NOT move tone marks, expand slang, lowercase,
strip emoji, or verbalize numbers. NFC only canonicalizes Unicode composition;
'hóa' and 'hoá' are different legitimate spellings, not NFC equivalents.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Mapping

# These are ambiguous informal forms. Enable only for a corpus that permits it.
OPTIONAL_TEENCODE: dict[str, str] = {
    "ko": "không",
    "khum": "không",
    "hok": "không",
    "dc": "được",
    "đc": "được",
    "j": "gì",
    "r": "rồi",
    "=))": "(cười)",
    ":))": "(cười)",
}
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_RESERVED = "<|endoftext|>"


@dataclass(frozen=True)
class CleanOptions:
    lowercase: bool = False
    strip_html: bool = False
    expand_teencode: bool = False
    replacements: Mapping[str, str] | None = None


def normalize_text(value: str, options: CleanOptions = CleanOptions()) -> str:
    """Normalize one story without altering its semantic content by default.

    - NFC is always applied, as is CRLF normalization and control cleanup.
    - Orthographic tone location is intentionally untouched.
    - Numbers, URL, email, punctuation and emoji remain by default.
    - Teencode uses whole-word matching, but may alter intended meaning.
    """
    if not isinstance(value, str):
        raise TypeError("Input must be a string")
    s = value.replace("\r\n", "\n").replace("\r", "\n")
    s = unicodedata.normalize("NFC", s)
    if options.strip_html:
        # Explicitly opt-in: prose containing markup-like text may be affected.
        s = unescape(_HTML_TAG.sub(" ", s))
    # Drop unsafe controls, preserving newlines and tabs; retain ZWJ/VS16 for emoji.
    s = "".join(ch for ch in s if (ch in "\n\t" or unicodedata.category(ch) != "Cc"))
    s = s.replace("\ufeff", "").replace("\u200b", "")
    if options.lowercase:
        s = s.lower()
    replacements: dict[str, str] = {}
    if options.expand_teencode:
        replacements.update(OPTIONAL_TEENCODE)
    if options.replacements:
        replacements.update(options.replacements)
    # Longest-first avoids corrupting overlapping phrases. No replacement by default.
    for original in sorted(replacements, key=len, reverse=True):
        replacement = replacements[original]
        if not original:
            raise ValueError("Empty replacement key")
        if not isinstance(replacement, str):
            raise TypeError(f"Replacement for {original!r} must be string")
        pat = re.compile(r"(?<!\w)" + re.escape(original) + r"(?!\w)", re.UNICODE | re.IGNORECASE)
        s = pat.sub(lambda _m, v=replacement: v, s)
    s = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in s.split("\n"))
    return unicodedata.normalize("NFC", s).strip()


def load_replacements(path: Path) -> dict[str, str]:
    """Load optional explicit spelling/teencode mapping from a JSON object."""
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Replacement dictionary must be a JSON object")
    if not all(isinstance(k, str) and isinstance(v, str) and k for k, v in data.items()):
        raise ValueError("Replacement dictionary must contain nonempty string keys and string values")
    if any(_RESERVED in k or _RESERVED in v for k, v in data.items()):
        raise ValueError("Reserved token cannot appear in replacement dictionary")
    return data
