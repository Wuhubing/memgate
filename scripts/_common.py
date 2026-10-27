"""Helpers shared by the entry-point scripts."""
from __future__ import annotations

from dotenv import load_dotenv

from memgate.config import load_llm
from memgate.experiment import resolve
from memgate.harness.llm_client import DeepSeekClient, FakeLLM


def make_llm(llm_config: str, world, fake: bool):
    """Real DeepSeek client (key from .env / environment), or the offline FakeLLM."""
    cfg = load_llm(resolve(llm_config))
    if fake:
        return FakeLLM(world), cfg
    load_dotenv(resolve(".env"))
    return DeepSeekClient(cfg), cfg
