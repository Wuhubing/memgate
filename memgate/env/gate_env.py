"""MemGateEnv: one episode = one session, one step = one turn (plan §3.7)."""
from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np

from memgate.config import RewardConfig
from memgate.env.features import OBS_DIM, featurize
from memgate.env.state import SessionState, advance_, new_state, note_candidates_
from memgate.harness.scorer import compute_reward
from memgate.retrieval.retriever import Retriever
from memgate.schema import Candidate, MemoryRecord, Mode, SessionScript

N_ACTIONS = 8


def decode_action(a: int, cands: list[Candidate | None]) -> list[MemoryRecord]:
    """Bit i of `a` injects candidate i; bits of empty slots are ignored (plan §2.2)."""
    return [c.record for i, c in enumerate(cands) if (int(a) >> i) & 1 and c is not None]


def canonical_action(a: int, cands: list[Candidate | None]) -> int:
    """Smallest equivalent action (E2): clears the bits of empty slots."""
    return sum(1 << i for i, c in enumerate(cands) if (int(a) >> i) & 1 and c is not None)


class StepInfo(dict):
    """Info dict that deepcopy leaves alone: SB3's DummyVecEnv deep-copies every info on every step,
    and these hold the session state and candidates (copying them made training ~100x slower)."""

    def __deepcopy__(self, memo):
        return self


class MemGateEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, world, backend, *, mode: Mode = "persistent", reward: RewardConfig,
                 scripts: list[SessionScript] | None = None, factory=None, T_choices: list[int] | None = None,
                 mask_history: bool = False, seed: int = 0, retriever: Retriever | None = None):
        super().__init__()
        assert scripts or factory, "need fixed scripts or a SessionFactory"
        self.world = world
        self.backend = backend
        self.mode = mode
        self.reward_cfg = reward
        self.scripts = scripts
        self.factory = factory
        self.T_choices = T_choices or [world.cfg.T_default]
        self.mask_history = mask_history
        self.retriever = retriever or Retriever(world.emb)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self._script_ptr = 0
        self.st: SessionState | None = None
        self.cands: list[Candidate | None] = []

    # ---- helpers -----------------------------------------------------------------------------
    def _next_script(self, options: dict | None) -> SessionScript:
        if options and "script" in options:
            return options["script"]
        if self.scripts is not None:
            if options and options.get("sequential"):
                s = self.scripts[self._script_ptr % len(self.scripts)]
                self._script_ptr += 1
                return s
            return self.scripts[int(self.rng.integers(len(self.scripts)))]
        return self.factory.make(int(self.rng.choice(self.T_choices)))

    def _retrieve(self) -> list[Candidate | None]:
        st = self.st
        exclude = set(st.in_context) if self.mode == "persistent" else set()
        cands = self.retriever.topk(st.turn.text, st.store, exclude)
        return cands

    def _obs(self) -> np.ndarray:
        return featurize(self.st, self.cands, self.mask_history)

    def info(self) -> dict:
        return StepInfo(cands=self.cands, state=self.st, mode=self.mode)

    # ---- gym API -----------------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:   # a seeded reset restarts every random stream, so it is reproducible
            self.rng = np.random.default_rng(seed)
            if hasattr(self.backend, "reseed"):
                self.backend.reseed(seed + 1)
            if self.factory is not None:
                self.factory.seed, self.factory.counter = seed, 0
        script = self._next_script(options)
        self.st = new_state(script, self.world.user(script.user_id).initial_store, self.mode)
        self.cands = self._retrieve()
        obs = self._obs()
        note_candidates_(self.st, [c.record.id for c in self.cands if c is not None])
        return obs, self.info()

    def step(self, action):
        st = self.st
        turn = st.turn
        chosen = decode_action(int(action), self.cands)
        res = self.backend.step(st, turn, chosen)
        out = compute_reward(st, turn, chosen, res.answer, res.mentioned, res.intrusions, self.reward_cfg,
                             res.hedged)
        advance_(st, turn, chosen, out, res.stored_user, res.response)
        terminated = st.done            # time is in the state: a true terminal, not a truncation (E1)
        if terminated:
            self.cands = [None, None, None]
            obs = np.zeros(OBS_DIM, dtype=np.float32)
        else:
            self.cands = self._retrieve()
            obs = self._obs()
            note_candidates_(st, [c.record.id for c in self.cands if c is not None])
        info = self.info()
        info.update(outcome=out, chosen=[r.id for r in chosen], turn=turn, step=res)
        return obs, float(out.reward), terminated, False, info

    def fork(self) -> MemGateEnv:
        return copy.deepcopy(self)
