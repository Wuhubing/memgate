"""Typed configuration. Every YAML goes through these models; unknown keys are errors."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PriceTable(Strict):
    input_cache_hit: float
    input_cache_miss: float
    output: float
    peak_multiplier: float = 2.0
    peak_hours: list[tuple[int, int]] = Field(default_factory=list)


class LLMConfig(Strict):
    base_url: str
    model: str
    api_key_env: str = "DEEPSEEK_API_KEY"
    disable_thinking: bool = True
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 160
    timeout_s: float = 120
    max_retries: int = 5
    concurrency: int = 8
    system_prompt_version: Literal["v1", "v2"] = "v1"
    cache_path: str = "cache/llm_cache.sqlite"
    budget_cny: float = 60.0
    price: PriceTable
    offpeak_only: bool = True


class SplitSizes(Strict):
    train: int
    val: int
    test: int


class WorldConfig(Strict):
    seed: int = 0
    n_users: int = 100
    split: SplitSizes
    n_changing: int = 6
    n_preference: int = 4
    n_trivia: int = 4
    p_stale: float = 0.5
    max_old_versions: int = 3
    spare_values: int = 12
    T_default: int = 12
    p_kind: dict[str, float]
    p_chat_topic: float = 0.5
    zipf_a: float = 1.0
    scripts_per_val_user: int = 50
    scripts_per_test_user: int = 100
    log_scripts: int = 3000
    log_T_choices: list[int] = Field(default_factory=lambda: [12])
    embedder: Literal["minilm", "hash"] = "minilm"


class RewardConfig(Strict):
    delta: float = 1.0
    alpha: float = 0.3
    beta: float = 1.5
    lam: float = 0.002


class RewardFile(Strict):
    default: str
    variants: dict[str, RewardConfig]


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_llm(path: str | Path = CONFIG_DIR / "llm.yaml") -> LLMConfig:
    return LLMConfig(**load_yaml(path))


def load_world(path: str | Path = CONFIG_DIR / "world.yaml", **overrides) -> WorldConfig:
    return WorldConfig(**{**load_yaml(path), **overrides})


def load_reward(name: str | None = None, path: str | Path = CONFIG_DIR / "reward.yaml") -> RewardConfig:
    rf = RewardFile(**load_yaml(path))
    return rf.variants[name or rf.default]


def config_hash(obj) -> str:
    """Stable short hash of any JSON-able config (pydantic models included)."""
    if isinstance(obj, BaseModel):
        obj = obj.model_dump()
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def save_run_metadata(run_dir: Path, name: str = "config.json", **configs) -> None:
    """Write the merged configs, the git commit and pip freeze next to the results."""
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {k: (v.model_dump() if isinstance(v, BaseModel) else v) for k, v in configs.items()}
    meta["git_commit"] = git_commit()
    (run_dir / name).write_text(json.dumps(meta, indent=2, default=str))
    try:
        freeze = subprocess.check_output(["pip", "freeze"], text=True)
        (run_dir / "pip_freeze.txt").write_text(freeze)
    except Exception:
        pass
