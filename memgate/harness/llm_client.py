"""LLM access: the DeepSeek client (cache, retries, budget, off-peak) and a FakeLLM for offline runs.

Plan §3.4 and problems A1-A14.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import random
import re
import sqlite3
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from memgate.config import ROOT, LLMConfig
from memgate.schema import LLMResponse, Usage
from memgate.text import sha


class LLMClient(Protocol):
    usage: Usage

    async def chat(self, messages: list[dict], *, session_id: str, turn_idx: int, sample_idx: int = 0,
                   meta: dict | None = None) -> LLMResponse: ...


class FatalLLMError(RuntimeError):
    """Stops the whole run: bad key, no balance, misconfiguration (A7)."""


class BudgetExceeded(FatalLLMError):
    pass


def cache_key(model: str, messages: list[dict], temperature: float, top_p: float, max_tokens: int,
              session_id: str, turn_idx: int, sample_idx: int) -> str:
    """Includes the session and turn so two sessions that reach identical messages still get
    independent samples (plan §3.4)."""
    blob = json.dumps([model, messages, temperature, top_p, max_tokens, session_id, turn_idx, sample_idx],
                      sort_keys=True, ensure_ascii=False)
    return sha(blob, 64)


class ResponseCache:
    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, body TEXT, created REAL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS spend (id INTEGER PRIMARY KEY AUTOINCREMENT, cost REAL, "
                        "created REAL, model TEXT)")
        self.db.commit()

    def get(self, key: str) -> LLMResponse | None:
        row = self.db.execute("SELECT body FROM responses WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        d = json.loads(row[0])
        d["from_cache"] = True
        return LLMResponse(**d)

    def put(self, key: str, resp: LLMResponse) -> None:
        d = {k: getattr(resp, k) for k in resp.__dataclass_fields__ if k != "from_cache"}
        self.db.execute("INSERT OR REPLACE INTO responses VALUES (?,?,?)", (key, json.dumps(d), time.time()))
        self.db.commit()

    def add_spend(self, cost: float, model: str) -> None:
        self.db.execute("INSERT INTO spend (cost, created, model) VALUES (?,?,?)", (cost, time.time(), model))
        self.db.commit()

    def total_spend(self) -> float:
        return float(self.db.execute("SELECT COALESCE(SUM(cost), 0) FROM spend").fetchone()[0])


def is_peak(cfg: LLMConfig, now_utc: dt.datetime | None = None) -> bool:
    now = (now_utc or dt.datetime.now(dt.UTC)).astimezone(dt.timezone(dt.timedelta(hours=8)))
    if now.weekday() >= 5:
        return False
    return any(a <= now.hour < b for a, b in cfg.price.peak_hours)


def seconds_until_offpeak(cfg: LLMConfig, now_utc: dt.datetime | None = None) -> float:
    now = now_utc or dt.datetime.now(dt.UTC)
    t = now
    for _ in range(24 * 12):
        if not is_peak(cfg, t):
            return (t - now).total_seconds()
        t += dt.timedelta(minutes=5)
    return 0.0


def call_cost(cfg: LLMConfig, hit: int, miss: int, completion: int, peak: bool) -> float:
    p = cfg.price
    c = (hit * p.input_cache_hit + miss * p.input_cache_miss + completion * p.output) / 1e6
    return c * (p.peak_multiplier if peak else 1.0)


class DeepSeekClient:
    def __init__(self, cfg: LLMConfig, *, dry_budget: bool = False):
        if "reasoner" in cfg.model:
            raise FatalLLMError("use the non-thinking chat model, not a reasoner (A2)")
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise FatalLLMError(f"environment variable {cfg.api_key_env} is not set (put it in .env)")
        self.cfg = cfg
        self._key = key
        self._loop: asyncio.AbstractEventLoop | None = None
        self.client = None
        self.sem: asyncio.Semaphore | None = None
        self.cache = ResponseCache(cfg.cache_path)
        self.usage = Usage()
        self.limit = cfg.concurrency
        self._shrinking = False
        self.dry_budget = dry_budget

    def _make_client(self):
        from openai import AsyncOpenAI
        return AsyncOpenAI(base_url=self.cfg.base_url, api_key=self._key, timeout=self.cfg.timeout_s, max_retries=0)

    def _bind_loop(self) -> None:
        """The HTTP connection pool and the semaphore belong to one event loop. Scripts that call
        asyncio.run several times (run_pilot: one per condition) get a fresh pair per loop; reusing
        the old pool raised 'Event loop is closed' on the first calls of every later loop."""
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            self._loop = loop
            self.client = self._make_client()
            self.sem = asyncio.Semaphore(self.limit)

    async def aclose(self) -> None:
        """Close the connection pool inside the loop that owns it. Left to the garbage collector,
        the pool's close runs after asyncio.run has shut the loop and logs 'Event loop is closed'."""
        if self.client is not None and self._loop is asyncio.get_running_loop():
            await self.client.close()
        self.client, self._loop = None, None

    async def chat(self, messages: list[dict], *, session_id: str, turn_idx: int, sample_idx: int = 0,
                   meta: dict | None = None) -> LLMResponse:
        cfg = self.cfg
        self._bind_loop()
        key = cache_key(cfg.model, messages, cfg.temperature, cfg.top_p, cfg.max_tokens, session_id, turn_idx,
                        sample_idx)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if not self.dry_budget and self.cache.total_spend() >= cfg.budget_cny:
            raise BudgetExceeded(f"budget of {cfg.budget_cny} CNY reached")
        async with self.sem:
            if cfg.offpeak_only:
                wait = seconds_until_offpeak(cfg)
                if wait > 0:
                    print(f"[llm] peak hours, waiting {wait / 60:.0f} min for off-peak pricing")
                    await asyncio.sleep(wait)
            resp = await self._call_with_retries(messages)
            if not resp.text.strip():   # A8: one extra try for an empty reply
                resp = await self._call_with_retries(messages)
        cost = call_cost(cfg, resp.cache_hit or 0, resp.cache_miss if resp.cache_miss is not None
                         else resp.prompt_tokens, resp.completion_tokens, is_peak(cfg))
        self.cache.put(key, resp)
        self.cache.add_spend(cost, resp.model)
        u = self.usage
        u.calls += 1
        u.prompt_tokens += resp.prompt_tokens
        u.completion_tokens += resp.completion_tokens
        u.cache_hit += resp.cache_hit or 0
        u.cache_miss += resp.cache_miss or 0
        u.cost_cny += cost
        return resp

    async def _call_with_retries(self, messages: list[dict]) -> LLMResponse:
        import openai
        cfg = self.cfg
        extra = {"thinking": {"type": "disabled"}} if cfg.disable_thinking else {}
        last = None
        for attempt in range(cfg.max_retries + 1):
            t0 = time.monotonic()
            try:
                r = await self.client.chat.completions.create(
                    model=cfg.model, messages=messages, temperature=cfg.temperature, top_p=cfg.top_p,
                    max_tokens=cfg.max_tokens, extra_body=extra)
                return self._parse(r, time.monotonic() - t0)
            except openai.RateLimitError as e:
                last = e
                self._halve_concurrency()
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as e:
                last = e
            except openai.APIStatusError as e:
                if e.status_code == 429:
                    last = e
                    self._halve_concurrency()
                elif e.status_code in (500, 502, 503, 504):
                    last = e
                else:   # 400 / 401 / 402 / 422: retrying cannot help (A7)
                    raise FatalLLMError(f"HTTP {e.status_code}: {e.message}") from e
            await asyncio.sleep(2 ** attempt + random.random())
        raise RuntimeError(f"LLM call failed after {cfg.max_retries + 1} attempts: {last!r}")

    def _halve_concurrency(self) -> None:
        """A5: on rate limiting, permanently take half of the semaphore's slots (never below 1)."""
        if self._shrinking or self.limit <= 1:
            return
        self._shrinking = True
        target = max(self.limit // 2, 1)

        async def take():
            for _ in range(self.limit - target):
                await self.sem.acquire()
            print(f"[llm] rate limited: concurrency {self.limit} -> {target}", flush=True)
            self.limit = target
            self._shrinking = False
        asyncio.get_running_loop().create_task(take())

    def _parse(self, r, latency: float) -> LLMResponse:
        ch = r.choices[0]
        u = r.usage
        hit = getattr(u, "prompt_cache_hit_tokens", None)
        miss = getattr(u, "prompt_cache_miss_tokens", None)
        reasoning = 0
        det = getattr(u, "completion_tokens_details", None)
        if det is not None:
            reasoning = getattr(det, "reasoning_tokens", 0) or 0
        if reasoning:
            raise FatalLLMError("reply contains reasoning tokens: thinking mode is on (A2)")
        return LLMResponse(
            text=ch.message.content or "", model=r.model, system_fingerprint=getattr(r, "system_fingerprint", None),
            prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens, cache_hit=hit, cache_miss=miss,
            reasoning_tokens=reasoning, finish_reason=ch.finish_reason, latency_s=latency)


# ---------------------------------------------------------------------------------------------
# FakeLLM: a rule-based stand-in with the behaviours the project studies (anchoring on injected
# old values, repeated unrelated mentions that persist while a memory stays in the history).
# Used by tests and for running the whole pipeline offline. Its numbers mean nothing about real LLMs.

_MEM_LINE = re.compile(r"^- (.+)$", re.M)


class FakeLLM:
    def __init__(self, world, *, p_anchor: float = 0.45, p_intrude: float = 0.12, p_topic: float = 0.25,
                 fail_sessions: set[str] | None = None):
        self.world = world
        self.by_text = {r.text: r for r in world.records.values()}
        self.p_anchor, self.p_intrude, self.p_topic = p_anchor, p_intrude, p_topic
        self.usage = Usage()
        self.fail_sessions = fail_sessions or set()

    async def chat(self, messages: list[dict], *, session_id: str, turn_idx: int, sample_idx: int = 0,
                   meta: dict | None = None) -> LLMResponse:
        if session_id in self.fail_sessions and turn_idx == 2:
            raise RuntimeError("simulated failure")
        turn, user = meta["turn"], self.world.user(meta["user_id"])
        rng = np.random.default_rng(int(sha(f"{session_id}|{turn_idx}|{sample_idx}", 12), 16))
        visible, age = [], {}
        for i, m in enumerate(messages):
            if m["role"] != "user":
                continue
            for line in _MEM_LINE.findall(m["content"]):
                r = self.by_text.get(line)
                if r is not None and r.id not in age:
                    visible.append(r)
                    age[r.id] = sum(1 for x in messages[i:] if x["role"] == "user")
        said_by_user = user.index.values_in(" ".join(m["content"] for m in messages[1:-1] if m["role"] == "user"))
        said_by_me = user.index.values_in(" ".join(m["content"] for m in messages if m["role"] == "assistant"))
        parts = []
        if turn.kind == "query":
            cur_vis = any(r.value == turn.answer_value for r in visible) or turn.answer_value in said_by_user
            stale_vis = [r for r in visible if r.value in turn.stale_values]
            if stale_vis and rng.random() < (0.85 if not cur_vis else self.p_anchor + 0.05 * max(age.values())):
                parts.append(f"It is {stale_vis[0].value}.")
            elif cur_vis and rng.random() < 0.92:
                parts.append(f"It is {turn.answer_value}.")
            else:
                parts.append("I don't have that information yet.")
        elif turn.kind == "update":
            parts.append("Got it, I'll remember that.")
        else:
            parts.append("Here is a suggestion: keep it short and friendly.")
        for r in visible:
            if r.key == turn.target_key:
                continue
            p = self.p_intrude + (self.p_topic if turn.topic_key == r.key else 0.0)
            p += 0.1 * (r.value in said_by_me)
            if rng.random() < p:
                parts.append(f"By the way, {r.text[0].lower() + r.text[1:]}")
        for v in said_by_me - {r.value for r in visible}:
            if rng.random() < 0.05:   # ephemeral path: old replies keep it alive
                parts.append(f"Also, remember {v}.")
        text = " ".join(parts)
        n_in = sum(len(m["content"].split()) for m in messages)
        self.usage.calls += 1
        self.usage.prompt_tokens += n_in
        self.usage.completion_tokens += len(text.split())
        return LLMResponse(text=text, model="fake", prompt_tokens=n_in, completion_tokens=len(text.split()),
                           cache_hit=0, cache_miss=n_in, finish_reason="stop")
