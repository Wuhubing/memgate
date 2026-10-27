"""Fictional value pools.

Every value is built from generated "core" words (pronounceable nonsense such as "Varnel") so the
model cannot answer from its own knowledge and cannot paraphrase a value (plan §3.1, B1/B2).
Cores are unique across the whole world; dates are unique within one user.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

ONSETS = ["b", "br", "c", "d", "dr", "f", "g", "gr", "h", "k", "l", "m", "n", "p", "qu", "r", "s",
          "st", "t", "tr", "v", "w", "z"]
VOWELS = ["a", "e", "i", "o", "u", "ai", "ei", "ou"]
CODAS = ["", "l", "n", "r", "s", "th", "nd", "rk", "x", "m"]

# Words a generator could stumble into that would make matching or meaning ambiguous.
BLACKLIST = {
    "dinner", "banner", "border", "boston", "bottom", "bread", "brick", "brother", "build", "button", "candle",
    "carbon", "center", "dragon", "drama", "father", "garden", "gold", "harbor", "honey", "lemon", "level",
    "linen", "london", "mason", "matter", "melon", "mirror", "model", "monday", "mother", "native", "nature",
    "paper", "parker", "pencil", "pilot", "planet", "potter", "rocket", "salmon", "silver", "sister", "stone",
    "summer", "table", "talon", "tender", "timber", "tower", "travel", "turner", "valley", "velvet", "water",
    "winter", "wonder", "zone", "room", "style", "notes", "air", "theme", "lane", "fitness", "coffee", "house",
    "project", "deadline", "contact", "friday", "saturday", "sunday", "tuesday", "wednesday", "thursday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "the", "and", "for", "with", "team", "cat", "gym", "flight", "editor", "writing", "note", "taking",
}

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
YEAR = 2026  # never written into any text


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


class CoreGen:
    """Draws pronounceable nonsense words, never repeating one within a world."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.used: set[str] = set()

    def draw(self) -> str:
        for _ in range(10_000):
            n_syl = int(self.rng.integers(2, 4))
            w = "".join(self.rng.choice(ONSETS) + self.rng.choice(VOWELS) for _ in range(n_syl))
            w += self.rng.choice(CODAS)
            w = w.capitalize()
            low = w.lower()
            if len(low) < 5 or len(low) > 11 or low in self.used or low in BLACKLIST:
                continue
            self.used.add(low)
            return w
        raise RuntimeError("core pool exhausted")


def date_value(day_of_year: int) -> tuple[str, list[str]]:
    """Value like 'Friday 14 November' plus every alias the scorer accepts (plan §3.1)."""
    d = dt.date(YEAR, 1, 1) + dt.timedelta(days=int(day_of_year))
    wd, m, ma, n = WEEKDAYS[d.weekday()], MONTHS[d.month - 1], MONTH_ABBR[d.month - 1], d.day
    value = f"{wd} {n} {m}"
    aliases = [value, f"{n} {m}", f"{m} {n}", f"{ma} {n}", f"{n} {ma}", f"{m} {ordinal(n)}",
               f"{ordinal(n)} {m}", f"{ordinal(n)} of {m}", f"{ma}. {n}", f"{wd}, {m} {n}", f"{wd}, {n} {m}"]
    return value, sorted(set(aliases))


class ValueFactory:
    """Builds (value, aliases) for every attribute type."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.cores = CoreGen(rng)

    def core(self) -> str:
        return self.cores.draw()

    def person(self) -> tuple[str, list[str]]:
        first, last = self.core(), self.core()
        return f"{first} {last}", [f"{first} {last}", last]

    def with_suffix(self, fmt: str) -> tuple[str, list[str]]:
        c = self.core()
        full = fmt.format(c)
        bare = full[4:] if full.startswith("the ") else full
        return full, sorted({bare, c})

    def draw(self, vtype: str, used_days: set[int] | None = None) -> tuple[str, list[str]]:
        if vtype == "date":
            assert used_days is not None
            while True:
                day = int(self.rng.integers(0, 365))
                if day not in used_days:
                    used_days.add(day)
                    return date_value(day)
        if vtype == "person":
            return self.person()
        fmt = {
            "style": "the {} style", "tool": "{} Notes", "airline": "{} Air", "theme": "the {} theme",
            "pet": "{}", "cafe": "{} Coffee House", "gym": "{} Fitness", "street": "{} Lane",
        }[vtype]
        return self.with_suffix(fmt)
