from __future__ import annotations

import re

from memgate.harness.prompt import SYSTEM_V1, build_messages, memory_block


def test_block_has_no_dates_or_version_labels(world):
    recs = list(world.split_users("train")[0].initial_store)[:3]
    block = memory_block(recs)
    assert block.startswith("<memory>") and all(r.text in block for r in recs)
    assert not re.search(r"\b(updated|old|new|version|created|day \d+)\b", block, re.I)
    assert memory_block([]) == ""


def test_order_is_the_given_order(world):
    recs = list(world.split_users("train")[0].initial_store)[:3]
    block = memory_block(recs[::-1])
    pos = [block.index(r.text) for r in recs[::-1]]
    assert pos == sorted(pos)


def test_persistent_vs_ephemeral_storage(world):
    r = world.split_users("train")[0].initial_store[0]
    msgs, stored = build_messages(SYSTEM_V1, [], "hello", [r], "persistent")
    assert msgs[0] == {"role": "system", "content": SYSTEM_V1}
    assert r.text in msgs[-1]["content"] and msgs[-1]["content"].endswith("hello")
    assert stored == msgs[-1]["content"]
    msgs_e, stored_e = build_messages(SYSTEM_V1, [], "hello", [r], "ephemeral")
    assert msgs_e == msgs           # what the model sees this turn is identical
    assert stored_e == "hello"      # but the block is not kept in history


def test_history_prefix_is_stable(world):
    """Each turn's messages extend the previous turn's history (prefix cache, D2)."""
    recs = list(world.split_users("train")[0].initial_store)
    for mode in ("persistent", "ephemeral"):
        hist, prev = [], None
        for t in range(4):
            msgs, stored = build_messages(SYSTEM_V1, hist, f"turn {t}", [recs[t]], mode)
            if prev is not None:
                assert msgs[: len(prev) - 1] == prev[:-1]
                if mode == "persistent":
                    assert msgs[: len(prev)] == prev
            hist = hist + [(stored, f"reply {t}")]
            prev = msgs
