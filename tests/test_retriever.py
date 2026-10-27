from __future__ import annotations

from memgate.retrieval.retriever import Retriever
from memgate.world.generator import SessionFactory


def test_always_k_slots_and_exclude(world):
    ret = Retriever(world.emb)
    store = list(world.split_users("train")[0].initial_store)
    c = ret.topk("What is my favourite airline?", store)
    assert len(c) == 3 and all(x is not None for x in c)
    assert [x.rank for x in c] == [1, 2, 3]
    assert c[0].fused >= c[1].fused >= c[2].fused
    ex = {x.record.id for x in c}
    c2 = ret.topk("What is my favourite airline?", store, ex)
    assert not ex & {x.record.id for x in c2 if x}
    few = ret.topk("anything", store[:2])
    assert few[2] is None and few[0] is not None
    assert ret.topk("anything", store, {r.id for r in store}) == [None, None, None]


def test_deterministic(world):
    ret1, ret2 = Retriever(world.emb), Retriever(world.emb)
    store = list(world.split_users("train")[1].initial_store)
    q = "Tell me about my week."
    assert [c.record.id for c in ret1.topk(q, store)] == [c.record.id for c in ret2.topk(q, store)]


def test_query_recall(world):
    """The record that answers a query is usually among the top 3 (the retriever is not the bottleneck)."""
    ret = Retriever(world.emb)
    fac = SessionFactory(world, "train", seed=4)
    hit = n = 0
    for _ in range(80):
        s = fac.make(12)
        store = list(world.user(s.user_id).initial_store)
        for t in s.turns:
            if t.kind == "update":
                store.append(t.new_record)
            elif t.kind == "query":
                n += 1
                hit += bool(t.relevant_ids & {c.record.id for c in ret.topk(t.text, store) if c})
    assert hit / n >= 0.6, hit / n
