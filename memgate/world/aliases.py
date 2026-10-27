"""Alias index: finds which memory values a piece of text mentions, with word-boundary matching."""
from __future__ import annotations

from collections import defaultdict

from memgate.text import normalize


class AliasIndex:
    def __init__(self, aliases: dict[str, list[str]]):
        """aliases: value -> list of surface forms (the value itself included)."""
        self.aliases = aliases
        self._by_first: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for value, forms in aliases.items():
            for f in {value, *forms}:
                n = normalize(f)
                toks = n.split()
                if toks:
                    self._by_first[toks[0]].append((n, value))

    def values_in(self, text: str, *, normalized: bool = False) -> set[str]:
        n = text if normalized else normalize(text)
        found: set[str] = set()
        for tok in set(n.split()):
            for alias_n, value in self._by_first.get(tok, ()):
                if value not in found and alias_n in n:
                    found.add(value)
        return found

    def hit(self, value: str, text: str, *, normalized: bool = False) -> bool:
        n = text if normalized else normalize(text)
        return any(normalize(f) in n for f in {value, *self.aliases.get(value, ())})
