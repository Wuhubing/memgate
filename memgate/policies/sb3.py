"""DQN / PPO training and loading with stable-baselines3 (plan §3.12, 原文档 模块 8)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from memgate.policies.heuristics import Policy

DQN_DEFAULTS = dict(learning_rate=3e-4, buffer_size=100_000, batch_size=256, target_update_interval=1_000,
                    exploration_fraction=0.3, exploration_final_eps=0.05, learning_starts=5_000,
                    net_arch=[128, 128])
PPO_DEFAULTS = dict(learning_rate=3e-4, n_steps=2048, batch_size=256, gae_lambda=0.95, ent_coef=0.01,
                    net_arch=[128, 128])


def _vec(make_env, n_envs: int):
    from stable_baselines3.common.vec_env import DummyVecEnv
    return DummyVecEnv([(lambda i=i: make_env(i)) for i in range(n_envs)])


def train_sb3(algo: str, make_env, *, gamma: float, total_steps: int, seed: int, run_dir: Path,
              n_envs: int = 8, eval_env=None, eval_freq: int = 10_000, overrides: dict | None = None):
    """make_env(i) builds sub-environment i; each gets its own seed (E7)."""
    import torch
    from stable_baselines3 import DQN, PPO
    torch.set_num_threads(1)   # E12: one thread per process, parallelism comes from processes
    kw = {**(DQN_DEFAULTS if algo == "dqn" else PPO_DEFAULTS), **(overrides or {})}
    net_arch = kw.pop("net_arch")
    env = _vec(make_env, n_envs)
    common = dict(gamma=gamma, seed=seed, verbose=0, tensorboard_log=None,
                  policy_kwargs=dict(net_arch=net_arch), device="cpu")
    if algo == "dqn":
        model = DQN("MlpPolicy", env, **common, **kw)
    elif algo == "ppo":
        model = PPO("MlpPolicy", env, **common, **kw)
    else:
        raise ValueError(algo)
    callback = None
    if eval_env is not None:
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor
        callback = EvalCallback(Monitor(eval_env), eval_freq=max(eval_freq // n_envs, 1), n_eval_episodes=50,
                                best_model_save_path=str(run_dir / "best"), log_path=str(run_dir / "eval"),
                                deterministic=True, verbose=0)
    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "policy.zip")
    return model


class SB3Policy(Policy):
    def __init__(self, path: Path, algo: str, name: str | None = None):
        from stable_baselines3 import DQN, PPO
        self.model = (DQN if algo == "dqn" else PPO).load(str(path), device="cpu")
        self.name = name or algo

    def act(self, obs, info=None) -> int:
        a, _ = self.model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
        return int(a)

    def q_values(self, obs) -> np.ndarray:
        import torch
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(obs, dtype=np.float32)[None])
            return self.model.q_net(t).numpy()[0]
