from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest

from memgate.config import load_llm
from memgate.harness.llm_client import (
    BudgetExceeded,
    DeepSeekClient,
    FatalLLMError,
    ResponseCache,
    cache_key,
    call_cost,
    is_peak,
    seconds_until_offpeak,
)
from memgate.schema import LLMResponse

MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]


@pytest.fixture
def cfg(tmp_path):
    base = load_llm()
    return base.model_copy(update={"cache_path": str(tmp_path / "c.sqlite"), "offpeak_only": False,
                                   "budget_cny": 1.0})


def test_cache_roundtrip_and_spend(tmp_path):
    c = ResponseCache(tmp_path / "x.sqlite")
    r = LLMResponse("hello", "m", prompt_tokens=3, completion_tokens=1, cache_hit=1, cache_miss=2)
    c.put("k", r)
    got = c.get("k")
    assert got.text == "hello" and got.from_cache and got.cache_miss == 2
    assert c.get("missing") is None
    c.add_spend(0.5, "m")
    c.add_spend(0.25, "m")
    assert ResponseCache(tmp_path / "x.sqlite").total_spend() == pytest.approx(0.75)   # persists


def test_cache_key_separates_samples():
    k = lambda **kw: cache_key("m", MSGS, 0.7, 1.0, 160, **{"session_id": "s", "turn_idx": 0,  # noqa: E731
                                                            "sample_idx": 0, **kw})
    assert k() == k()
    assert len({k(), k(session_id="t"), k(turn_idx=1), k(sample_idx=1)}) == 4


def test_peak_and_cost(cfg):
    bj = dt.timezone(dt.timedelta(hours=8))
    tue_10 = dt.datetime(2026, 10, 27, 10, tzinfo=bj)
    tue_23 = dt.datetime(2026, 10, 27, 23, tzinfo=bj)
    sat_10 = dt.datetime(2026, 10, 31, 10, tzinfo=bj)
    assert is_peak(cfg, tue_10) and not is_peak(cfg, tue_23) and not is_peak(cfg, sat_10)
    assert seconds_until_offpeak(cfg, tue_23) == 0
    assert 0 < seconds_until_offpeak(cfg, tue_10) <= 2 * 3600
    p = cfg.price
    assert call_cost(cfg, 1e6, 0, 0, False) == pytest.approx(p.input_cache_hit)
    assert call_cost(cfg, 0, 1e6, 1e6, True) == pytest.approx((p.input_cache_miss + p.output) * p.peak_multiplier)


def test_missing_key_is_fatal(cfg, monkeypatch):
    monkeypatch.delenv(cfg.api_key_env, raising=False)
    with pytest.raises(FatalLLMError):
        DeepSeekClient(cfg)


def fake_completion(text="It is fine.", reasoning=0):
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10, prompt_cache_hit_tokens=60,
                            prompt_cache_miss_tokens=40,
                            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning))
    ch = SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")
    return SimpleNamespace(choices=[ch], usage=usage, model="deepseek-flash", system_fingerprint="fp")


def client_with(cfg, monkeypatch, replies):
    monkeypatch.setenv(cfg.api_key_env, "test-not-a-key")
    c = DeepSeekClient(cfg)
    calls = []

    async def create(**kw):
        calls.append(kw)
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    c._make_client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return c, calls


async def test_call_cache_and_thinking_disabled(cfg, monkeypatch):
    c, calls = client_with(cfg, monkeypatch, [fake_completion()])
    r1 = await c.chat(MSGS, session_id="s", turn_idx=0)
    r2 = await c.chat(MSGS, session_id="s", turn_idx=0)
    assert len(calls) == 1 and r2.from_cache and r1.text == r2.text
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert calls[0]["top_p"] == cfg.top_p
    assert c.usage.calls == 1 and c.usage.cache_hit == 60
    assert c.cache.total_spend() == pytest.approx(call_cost(cfg, 60, 40, 10, is_peak(cfg)))


async def test_reasoning_tokens_are_fatal(cfg, monkeypatch):
    c, _ = client_with(cfg, monkeypatch, [fake_completion(reasoning=5)])
    with pytest.raises(FatalLLMError):
        await c.chat(MSGS, session_id="s", turn_idx=1)


async def test_empty_reply_retried_once(cfg, monkeypatch):
    c, calls = client_with(cfg, monkeypatch, [fake_completion(""), fake_completion("ok")])
    r = await c.chat(MSGS, session_id="s", turn_idx=2)
    assert r.text == "ok" and len(calls) == 2


async def test_budget_stops_new_calls(cfg, monkeypatch):
    c, calls = client_with(cfg, monkeypatch, [fake_completion()])
    c.cache.add_spend(cfg.budget_cny, "m")
    with pytest.raises(BudgetExceeded):
        await c.chat(MSGS, session_id="s", turn_idx=3)
    assert not calls


async def test_http_400_is_fatal_and_not_retried(cfg, monkeypatch):
    import openai
    err = openai.BadRequestError.__new__(openai.BadRequestError)   # no HTTP stack needed
    err.status_code, err.message = 400, "bad request"
    c, calls = client_with(cfg, monkeypatch, [err])
    with pytest.raises(FatalLLMError):
        await c.chat(MSGS, session_id="s", turn_idx=4)
    assert len(calls) == 1


def test_reasoner_model_rejected(cfg):
    with pytest.raises(FatalLLMError):
        DeepSeekClient(cfg.model_copy(update={"model": "deepseek-reasoner"}))


async def test_rate_limit_halves_concurrency(cfg, monkeypatch):
    import asyncio

    import openai
    err = openai.RateLimitError.__new__(openai.RateLimitError)
    err.status_code, err.message = 429, "slow down"
    monkeypatch.setattr("memgate.harness.llm_client.asyncio.sleep", _no_sleep)
    c, calls = client_with(cfg.model_copy(update={"concurrency": 8}), monkeypatch, [err, fake_completion()])
    r = await c.chat(MSGS, session_id="s", turn_idx=5)
    for _ in range(20):
        await asyncio.get_running_loop().run_in_executor(None, lambda: None)
        if c.limit == 4:
            break
    assert r.text and len(calls) == 2 and c.limit == 4


async def _no_sleep(*a, **k):
    return None


def test_client_rebuilt_per_event_loop(cfg, monkeypatch):
    """run_pilot calls asyncio.run once per condition; every loop must get its own HTTP client."""
    import asyncio
    c, calls = client_with(cfg, monkeypatch, [fake_completion(), fake_completion()])
    made = []
    orig = c._make_client
    c._make_client = lambda: made.append(1) or orig()
    asyncio.run(c.chat(MSGS, session_id="a", turn_idx=0))
    first = c.client
    asyncio.run(c.chat(MSGS, session_id="b", turn_idx=0))
    assert len(made) == 2 and c.client is not first and len(calls) == 2
