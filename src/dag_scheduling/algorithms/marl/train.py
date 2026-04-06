"""
MARL training script — chapter 1 of the thesis.

Hyperparameters (Table / Section MARL_train in chapter 1):
  Network:    FC 128→64→32 per agent
  Algorithm:  PPO (independent, no parameter sharing)
  gamma:      1.0  (finite-horizon, sparse terminal reward)
  lr:         2e-5 (Adam)
  iterations: 350_000
  rollouts/iter: 4 episodes

Usage:
  uv run python -m dag_scheduling.algorithms.marl.train \
      --ws 1 --n 30 --iters 350000 --out results/marl_ws1_n30
"""

from __future__ import annotations
import argparse
import random

import ray
from ray.rllib.algorithms.ppo import PPOConfig

from dag_scheduling.data.generator import generate
from dag_scheduling.data.augmentor import augment_random
from dag_scheduling.core.platform import make_workspace
from dag_scheduling.env.offline_env import OfflineSchedulingEnv, AGENT_IDS

# baseline parameter grid (Table in problem_statement, fat=0.5 fixed)
DENSITY   = [0.1, 0.4, 0.8]
REGULARITY = [0.2, 0.8]
JUMP      = [2, 4]
CCR       = [0.2, 0.8]
N_TRAIN_PER_CLASS = 3    # 24 classes × 3 = 72 training DAGs per size
N_TEST_PER_CLASS  = 20   # 24 classes × 20 = 480 test DAGs per size


def make_corpus(n: int, ws: int, n_per_class: int, seed_offset: int = 0):
    """Generate a corpus of (dag, platform) pairs covering all topology classes."""
    platform = make_workspace(ws)
    corpus = []
    idx = seed_offset
    for d in DENSITY:
        for r in REGULARITY:
            for j in JUMP:
                for c in CCR:
                    for _ in range(n_per_class):
                        dag = generate(n=n, fat=0.5, regular=r, density=d, jump=j, ccr=int(c * 10))
                        augment_random(dag, seed=idx)
                        corpus.append((dag, platform))
                        idx += 1
    return corpus


class CorpusEnv(OfflineSchedulingEnv):
    """Wraps OfflineSchedulingEnv to sample from a fixed corpus on each reset."""

    def __init__(self, env_config=None):
        super().__init__(env_config)
        cfg = env_config or {}
        self._corpus = cfg.get("corpus", [])

    def reset(self, *, seed=None, options=None):
        if self._corpus:
            dag, platform = random.choice(self._corpus)
            options = {"dag": dag, "platform": platform}
        return super().reset(seed=seed, options=options)


def build_config(ws: int, n: int, iters: int, num_gpus: int = 0) -> PPOConfig:
    train_corpus = make_corpus(n, ws, N_TRAIN_PER_CLASS, seed_offset=0)

    # FC network: 3 hidden layers [128, 64, 32] per agent (chapter 1)
    model_cfg = {
        "fcnet_hiddens": [128, 64, 32],
        "fcnet_activation": "relu",
    }

    policies = {a: (None, None, None, {"model": model_cfg}) for a in AGENT_IDS}

    config = (
        PPOConfig()
        .environment(
            env=CorpusEnv,
            env_config={"corpus": train_corpus},
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *args, **kw: agent_id,
            policies_to_train=list(AGENT_IDS),
        )
        .training(
            gamma=1.0,          # finite-horizon, sparse terminal reward
            lr=2e-5,
            # 4 trajectories per iteration (thesis §MARL_train)
            train_batch_size_per_learner=4,
            minibatch_size=4,
        )
        .framework("torch")
        .env_runners(num_env_runners=1)
        .resources(num_gpus=num_gpus)
    )
    return config


def _detect_gpus() -> int:
    """Return number of CUDA GPUs available, 0 if none."""
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def train(
    ws: int = 1,
    n: int = 30,
    iters: int = 350_000,
    out: str = "results/marl",
    num_gpus: int | None = None,
    log_interval: int = 1000,
):
    gpus = _detect_gpus() if num_gpus is None else num_gpus
    print(f"GPUs: {gpus}")
    ray.init(ignore_reinit_error=True)

    config = build_config(ws=ws, n=n, iters=iters, num_gpus=gpus)
    algo = config.build()

    from pathlib import Path
    import numpy as np
    reward_history: list[float] = []

    for i in range(iters):
        result = algo.train()
        if (i + 1) % log_interval == 0:
            mean_rew = result.get("env_runners", {}).get("episode_reward_mean", float("nan"))
            ep_len   = result.get("env_runners", {}).get("episode_len_mean",    float("nan"))
            print(f"iter {i+1:>7}/{iters}  reward={mean_rew:.4f}  ep_len={ep_len:.1f}")
            reward_history.append(mean_rew)

    path = algo.save(out)
    log_path = str(Path(out) / "reward_history.npy")
    Path(out).mkdir(parents=True, exist_ok=True)
    np.save(log_path, np.array(reward_history))
    print(f"saved checkpoint → {path}")
    print(f"saved reward log  → {log_path}")
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ws",       type=int,  default=1)
    p.add_argument("--n",        type=int,  default=30)
    p.add_argument("--iters",    type=int,  default=350_000)
    p.add_argument("--out",      type=str,  default="results/marl")
    p.add_argument("--gpus",     type=int,  default=None,
                   help="number of GPUs (default: auto-detect)")
    p.add_argument("--log_every", type=int, default=1000)
    args = p.parse_args()
    train(ws=args.ws, n=args.n, iters=args.iters, out=args.out,
          num_gpus=args.gpus, log_interval=args.log_every)
