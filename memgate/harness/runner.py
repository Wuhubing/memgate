"""Runs batches of sessions against a real (or fake) LLM and writes per-turn logs (plan §3.8)."""
from __future__ import annotations

import asyncio
import json
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from memgate.config import RewardConfig, config_hash, git_commit
from memgate.env.backends import result_from_reply
from memgate.env.features import featurize
from memgate.env.gate_env import decode_action
from memgate.env.state import SessionState, advance_, new_state, note_candidates_, snapshot_json
from memgate.harness.llm_client import FatalLLMError
from memgate.harness.prompt import SYSTEMS, build_messages
from memgate.harness.scorer import compute_reward
from memgate.retrieval.retriever import Retriever
from memgate.schema import Candidate, MemoryRecord, Mode, SessionScript, Turn, turn_to_dict

# chooser(state, turn, candidates, obs, info) -> records to inject
Chooser = Callable[[SessionState, Turn, list[Candidate | None], object, dict], list[MemoryRecord]]


def policy_chooser(policy, mask_history: bool = False) -> Chooser:
    def choose(st, turn, cands, obs, info):
        return decode_action(policy.act(obs, info), cands)
    return choose


async def run_session(script: SessionScript, chooser: Chooser, llm, world, *, mode: Mode, reward: RewardConfig,
                      retriever: Retriever, prompt_version: str = "v1", sample_idx: int = 0,
                      mask_history: bool = False, tags: dict | None = None) -> list[dict]:
    """One session, turn by turn. Returns its log rows; raises if any turn fails (no half sessions)."""
    st = new_state(script, world.user(script.user_id).initial_store, mode)
    system = SYSTEMS[prompt_version]
    rows = []
    for turn in script.turns:
        exclude = set(st.in_context) if mode == "persistent" else set()
        cands = retriever.topk(turn.text, st.store, exclude)
        obs = featurize(st, cands, mask_history)
        note_candidates_(st, [c.record.id for c in cands if c is not None])
        snap = snapshot_json(st)    # the state the outcome is produced from, as in MemGateEnv.step
        info = {"cands": cands, "state": st, "mode": mode}
        chosen = chooser(st, turn, cands, obs, info)
        msgs, stored = build_messages(system, st.history, turn.text, chosen, mode)
        resp = await llm.chat(msgs, session_id=script.id, turn_idx=turn.idx, sample_idx=sample_idx,
                              meta={"turn": turn, "user_id": script.user_id})
        res = result_from_reply(world, st, turn, resp.text, stored, resp)
        out = compute_reward(st, turn, chosen, res.answer, res.mentioned, res.intrusions, reward, res.hedged)
        cand_ids = [c.record.id if c else None for c in cands]
        action = sum(1 << i for i, c in enumerate(cands) if c is not None and c.record in chosen)
        rows.append({
            "session_id": script.id, "user_id": script.user_id, "t": turn.idx, "T": script.T, "kind": turn.kind,
            "mode": mode, "prompt_version": prompt_version, "sample_idx": sample_idx,
            "candidate_ids": json.dumps(cand_ids),
            "candidate_raw": json.dumps([[c.cos, c.bm25, c.rank, c.fused] if c else None for c in cands]),
            "obs": json.dumps([round(float(v), 6) for v in obs]),
            "action": action, "injected_ids": json.dumps([r.id for r in chosen]),
            "state_json": snap, "turn_json": json.dumps(turn_to_dict(turn)),
            "response": resp.text, "answer": out.answer, "correct": out.answer == "correct",
            "stale_used": out.answer == "stale", "hedged": out.hedged,
            "mentioned_ids": json.dumps(sorted(out.mentioned_ids)),
            "intrusion_ids": json.dumps(sorted(out.intrusion_ids)),
            "r_quality": out.r_quality, "r_stale": out.r_stale, "r_intrusion": out.r_intrusion,
            "r_tokens": out.r_tokens, "reward": out.reward,
            "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
            "prompt_cache_hit_tokens": resp.cache_hit, "prompt_cache_miss_tokens": resp.cache_miss,
            "finish_reason": resp.finish_reason, "llm_model": resp.model,
            "system_fingerprint": resp.system_fingerprint, "latency_s": resp.latency_s,
            "from_cache": resp.from_cache, "created_at_utc": time.time(),
            **(tags or {}),
        })
        advance_(st, turn, chosen, out, stored, resp.text)
    return rows


class ShardWriter:
    """Writes one parquet file per `every` sessions; never appends to an existing file (plan §3.8)."""

    def __init__(self, out_dir: Path, every: int = 100):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.every = every
        self.buf: list[dict] = []
        self.n_sessions = 0
        self.part = len(list(self.out_dir.glob("part-*.parquet")))

    def done_ids(self) -> set[str]:
        ids: set[str] = set()
        for p in self.out_dir.glob("part-*.parquet"):
            ids |= set(pd.read_parquet(p, columns=["session_id"])["session_id"].unique())
        return ids

    def add_session(self, rows: list[dict]) -> None:
        self.buf.extend(rows)
        self.n_sessions += 1
        if self.n_sessions % self.every == 0:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        pd.DataFrame(self.buf).to_parquet(self.out_dir / f"part-{self.part:05d}.parquet", index=False)
        self.part += 1
        self.buf = []


def read_logs(path: Path) -> pd.DataFrame:
    path = Path(path)
    parts = sorted(path.glob("**/part-*.parquet")) if path.is_dir() else [path]
    if not parts:
        raise FileNotFoundError(f"no log shards under {path}")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


async def run_sessions(scripts: list[SessionScript], make_chooser: Callable[[SessionScript], Chooser], llm, world, *,
                       mode: Mode, reward: RewardConfig, out_dir: Path, prompt_version: str = "v1",
                       concurrency: int = 8, tags_for: Callable[[SessionScript], dict] | None = None,
                       mask_history: bool = False, run_tags: dict | None = None, every: int = 100,
                       progress: bool = True) -> dict:
    writer = ShardWriter(out_dir, every)
    done = writer.done_ids()
    todo = [s for s in scripts if s.id not in done]
    retriever = Retriever(world.emb)
    sem = asyncio.Semaphore(concurrency)
    stats = {"done": 0, "failed": 0, "skipped": len(scripts) - len(todo)}
    fatal: list[BaseException] = []
    fail_log = open(Path(out_dir) / "failed.jsonl", "a")
    base_tags = {"git_commit": git_commit(), "config_hash": config_hash(reward), **(run_tags or {})}
    t0 = time.monotonic()

    async def one(s: SessionScript):
        if fatal:
            return
        async with sem:
            if fatal:
                return
            try:
                tags = {**base_tags, **(tags_for(s) if tags_for else {})}
                rows = await run_session(s, make_chooser(s), llm, world, mode=mode, reward=reward,
                                         retriever=retriever, prompt_version=prompt_version,
                                         mask_history=mask_history, tags=tags)
                writer.add_session(rows)
                stats["done"] += 1
            except FatalLLMError as e:
                fatal.append(e)
            except Exception as e:  # a failed session is dropped whole and recorded
                stats["failed"] += 1
                fail_log.write(json.dumps({"session_id": s.id, "error": repr(e),
                                           "trace": traceback.format_exc(limit=3)}) + "\n")
                fail_log.flush()
            if progress and (stats["done"] + stats["failed"]) % 20 == 0:
                u = llm.usage
                hit = u.cache_hit / max(u.cache_hit + u.cache_miss, 1)
                print(f"[runner] {stats['done']} ok, {stats['failed']} failed, {u.prompt_tokens} in / "
                      f"{u.completion_tokens} out tokens, {u.cost_cny:.2f} CNY, cache hit {hit:.0%}, "
                      f"{time.monotonic() - t0:.0f}s", flush=True)

    try:
        await asyncio.gather(*(one(s) for s in todo))
    finally:
        if hasattr(llm, "aclose"):
            await llm.aclose()
    writer.flush()
    fail_log.close()
    if fatal:
        raise fatal[0]
    return stats
