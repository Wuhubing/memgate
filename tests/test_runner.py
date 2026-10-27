from __future__ import annotations

import asyncio
import json

import pandas as pd

from memgate.harness.llm_client import FakeLLM
from memgate.harness.runner import policy_chooser, read_logs, run_sessions
from memgate.policies.heuristics import RandomPolicy
from memgate.world.generator import fixed_scripts

REQUIRED = {"session_id", "user_id", "t", "T", "kind", "mode", "candidate_ids", "obs", "action", "injected_ids",
            "state_json", "turn_json", "response", "answer", "mentioned_ids", "intrusion_ids", "r_quality",
            "r_stale", "r_intrusion", "r_tokens", "reward", "prompt_tokens", "completion_tokens",
            "prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "llm_model", "git_commit", "config_hash"}


def _run(world, reward, scripts, out, llm):
    return asyncio.run(run_sessions(scripts, lambda s: policy_chooser(RandomPolicy(0)), llm, world,
                                    mode="persistent", reward=reward, out_dir=out, every=3, progress=False))


def test_logs_failures_and_resume(world, reward, tmp_path):
    scripts = fixed_scripts(world, "train", 1, 6, seed=3)
    bad = {scripts[1].id}
    stats = _run(world, reward, scripts, tmp_path, FakeLLM(world, fail_sessions=bad))
    assert stats == {"done": len(scripts) - 1, "failed": 1, "skipped": 0}
    df = read_logs(tmp_path)
    assert REQUIRED <= set(df.columns)
    assert scripts[1].id not in set(df.session_id)               # failed sessions are dropped whole
    assert df.groupby("session_id").size().eq(6).all()
    fails = [json.loads(x) for x in (tmp_path / "failed.jsonl").read_text().splitlines()]
    assert [f["session_id"] for f in fails] == [scripts[1].id]
    # resume: only the failed one runs again
    stats2 = _run(world, reward, scripts, tmp_path, FakeLLM(world))
    assert stats2 == {"done": 1, "failed": 0, "skipped": len(scripts) - 1}
    df2 = read_logs(tmp_path)
    assert set(df2.session_id) == {s.id for s in scripts}
    assert not df2.duplicated(["session_id", "t"]).any()
    assert len(list(tmp_path.glob("part-*.parquet"))) >= 2       # never appended to an old shard


def test_reward_columns_add_up(world, reward, tmp_path):
    scripts = fixed_scripts(world, "train", 1, 5, seed=4)
    _run(world, reward, scripts, tmp_path, FakeLLM(world))
    df = read_logs(tmp_path)
    s = df.r_quality + df.r_stale + df.r_intrusion + df.r_tokens
    assert (s - df.reward).abs().max() < 1e-9
    assert pd.api.types.is_bool_dtype(df.correct)
