"""Text helpers shared by the generator, retriever and scorer."""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def normalize(s: str) -> str:
    """Lowercase, drop punctuation and markdown, collapse whitespace, pad with one space each side.

    Padding makes `" " + alias + " " in normalize(text)` a word-boundary match (plan §3.6, C2).
    """
    s = s.lower().replace("’", "'").replace("'s ", " ").replace("-", " ").replace("/", " ")
    s = _NON_ALNUM.sub(" ", s)
    return " " + _WS.sub(" ", s).strip() + " "


def norm_alias(a: str) -> str:
    return normalize(a)


def contains(norm_text: str, norm_alias_: str) -> bool:
    return norm_alias_ in norm_text


def tokenize(s: str) -> list[str]:
    return normalize(s).split()


def sha(s: str, n: int = 16) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:n]


@lru_cache(maxsize=1)
def _enc():
    """cl100k_base, or None when its BPE file cannot be downloaded (offline machines)."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(s: str) -> int:
    """Proxy token counter used for the λ term (plan D5). Counts are frozen into the world at
    generation time, so a world built offline (word-based fallback) stays self-consistent."""
    enc = _enc()
    if enc is None:
        return max(1, round(len(s.split()) * 1.3))
    return len(enc.encode(s))
