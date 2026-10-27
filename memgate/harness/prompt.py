"""Message assembly for the two injection modes (plan §3.5, D2)."""
from __future__ import annotations

from memgate.schema import MemoryRecord, Mode

SYSTEM_V1 = ("You are a helpful personal assistant. Notes about the user may appear inside "
             "<memory> tags. Use a note only if it is relevant to the user's current request. "
             "Answer in at most 80 words.")
SYSTEM_V2 = ("You are a helpful personal assistant. Notes about the user may appear inside "
             "<memory> tags. Answer in at most 80 words.")
SYSTEMS = {"v1": SYSTEM_V1, "v2": SYSTEM_V2}


def memory_block(chosen: list[MemoryRecord]) -> str:
    """No dates, versions or 'updated' labels (M3). Order = fused rank order of `chosen`."""
    if not chosen:
        return ""
    return "<memory>\n" + "\n".join(f"- {r.text}" for r in chosen) + "\n</memory>\n\n"


def build_messages(system: str, history: list[tuple[str, str]], user_text: str, chosen: list[MemoryRecord],
                   mode: Mode) -> tuple[list[dict], str]:
    """Returns (messages to send, user content to store in history).

    The memory block sits in front of the current user message in both modes, so the only
    difference is whether it stays in the history afterwards (persistent) or not (ephemeral).
    """
    current = memory_block(chosen) + user_text
    msgs = [{"role": "system", "content": system}]
    for u, a in history:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": current})
    stored = current if mode == "persistent" else user_text
    return msgs, stored
