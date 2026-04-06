"""
RLlib MultiAgentEnv for offline (full-visibility) MARL DAG scheduling.

Exactly as specified in chapter 1, Section 3 (MARL_env) of the thesis:

Agents:       one per executor type — "cpu", "gpu", "io"
Observation:  top-K=8 ready tasks × 13 metrics = 104-dim float vector
              sorted by agent's current active priority rule;
              padded with -1 if fewer than K tasks are ready.
              At episode start the ordering is topological.
Action:       Discrete(10) — 9 metric-based priority rules + no-op (9)
Reward:       0 during construction; terminal reward for agent i =
              -(AFT of the last task of type i to finish)
Termination:  per-agent when all tasks of that type are scheduled;
              __all__ when every task in the DAG is scheduled.

env_config keys:
  dag        SchedulingDAG (required)
  platform   Platform      (required)
"""

from __future__ import annotations
import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from dag_scheduling.core.dag import SchedulingDAG
from dag_scheduling.core.platform import Platform
from dag_scheduling.core.simulator import ScheduleState
from dag_scheduling.core.metrics import (
    compute_metrics, normalise,
    PRIORITY_RULES, NO_OP_ACTION, N_METRICS,
)

K = 8                               # observation cap (thesis §MARL_env)
OBS_DIM = K * N_METRICS             # 104
AGENT_IDS = ["cpu", "gpu", "io"]
TYPE_OF = {"cpu": "CPU", "gpu": "GPU", "io": "IO"}
AGENT_OF = {"CPU": "cpu", "GPU": "gpu", "IO": "io"}


class OfflineSchedulingEnv(MultiAgentEnv):
    """
    Offline DAG scheduling environment — matches Algorithm 1 of the thesis.

    Call reset(options={"dag": dag, "platform": platform}) to load a new
    instance, or supply defaults via env_config.
    """

    def __init__(self, env_config: dict | None = None):
        super().__init__()
        cfg = env_config or {}
        self._default_dag: SchedulingDAG | None = cfg.get("dag")
        self._default_platform: Platform | None = cfg.get("platform")

        self._possible_agents = AGENT_IDS
        self.agents = list(AGENT_IDS)

        obs_space = spaces.Box(
            low=-1.0, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        act_space = spaces.Discrete(NO_OP_ACTION + 1)   # 10

        self.observation_space = obs_space
        self.action_space = act_space

        # set per-agent spaces (required by newer RLlib)
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True
        self.observation_spaces = {a: obs_space for a in AGENT_IDS}
        self.action_spaces = {a: act_space for a in AGENT_IDS}

        # runtime state (initialised in reset)
        self._dag: SchedulingDAG | None = None
        self._platform: Platform | None = None
        self._state: ScheduleState | None = None
        self._metrics: np.ndarray | None = None     # (max_idx+1, 13), normalised
        self._topo_rank: dict[int, int] = {}        # idx -> topological position
        self._active_rule: dict[str, int] = {}      # agent -> rule index or -1 (topo)
        self._terminated: dict[str, bool] = {}
        self._type_tasks: dict[str, set[int]] = {}  # agent -> set of all task indices

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        options = options or {}
        dag = options.get("dag", self._default_dag)
        platform = options.get("platform", self._default_platform)
        if dag is None or platform is None:
            raise ValueError("dag and platform must be provided via options or env_config")

        self._dag = dag
        self._platform = platform
        self._state = ScheduleState(dag, platform)

        # precompute metrics once per episode, then normalise
        M_raw = compute_metrics(dag, platform)
        self._metrics = normalise(M_raw).astype(np.float32)

        # topological rank for initial ordering
        topo = dag.topological_order()
        self._topo_rank = {idx: pos for pos, idx in enumerate(topo)}

        # per-agent task sets
        self._type_tasks = {
            a: {idx for idx in dag.indices() if dag.node_type(idx) == TYPE_OF[a]}
            for a in AGENT_IDS
        }

        # initial rule: -1 = topological order
        self._active_rule = {a: -1 for a in AGENT_IDS}
        self._terminated = {a: False for a in AGENT_IDS}
        self.agents = [a for a in AGENT_IDS if self._type_tasks[a]]

        obs = {a: self._build_obs(a) for a in self.agents}
        return obs, {}

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action_dict: dict):
        """
        Process one round of actions (one iteration of the outer while-loop
        in Algorithm 1).  Agents act sequentially in CPU→GPU→IO order.
        """
        rew  = {a: 0.0 for a in self.agents}
        term = {a: False for a in self.agents}
        trunc = {a: False for a in self.agents}

        for agent in AGENT_IDS:
            if self._terminated.get(agent):
                continue
            if agent not in action_dict:
                continue

            action = int(action_dict[agent])
            ready_of_type = self._ready_for(agent)

            # no-op or empty queue → skip
            if action == NO_OP_ACTION or not ready_of_type:
                continue

            # update active rule (m_τ ← a_τ), then schedule top task
            self._active_rule[agent] = action
            task_idx = self._top_task(ready_of_type, action)
            self._state.schedule_task(task_idx)

            # check if this agent's type is fully done
            done_tasks = self._type_tasks[agent] & self._state.scheduled
            if done_tasks == self._type_tasks[agent]:
                # terminal reward = - finish time of last task of this type
                last_finish = max(
                    self._state.aft[t] for t in self._type_tasks[agent]
                )
                rew[agent] = -last_finish
                term[agent] = True
                self._terminated[agent] = True

        # global done when every task is scheduled
        all_done = self._state.is_done()
        term["__all__"] = all_done
        trunc["__all__"] = False

        # only agents that still have unfinished tasks should act next
        active = [a for a in self.agents if not self._terminated.get(a)]
        self.agents = active

        obs = {}
        if not all_done:
            obs = {a: self._build_obs(a) for a in self.agents}

        return obs, rew, term, trunc, {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ready_for(self, agent: str) -> list[int]:
        """Ready task indices whose node_type matches this agent."""
        node_type = TYPE_OF[agent]
        return [t for t in self._state.ready if self._dag.node_type(t) == node_type]

    def _top_task(self, ready: list[int], rule_idx: int) -> int:
        """Select top task from ready list using the given priority rule."""
        metric_idx, higher_is_better = PRIORITY_RULES[rule_idx]
        sign = -1.0 if higher_is_better else 1.0
        return min(ready, key=lambda t: sign * self._metrics[t, metric_idx])

    def _sort_ready(self, ready: list[int], rule_idx: int) -> list[int]:
        """Sort ready tasks by priority rule (descending = best first)."""
        if rule_idx == -1:
            # topological order
            return sorted(ready, key=lambda t: self._topo_rank.get(t, 0))
        metric_idx, higher_is_better = PRIORITY_RULES[rule_idx]
        return sorted(
            ready,
            key=lambda t: self._metrics[t, metric_idx],
            reverse=higher_is_better,
        )

    def _build_obs(self, agent: str) -> np.ndarray:
        """
        Build the 104-dim observation for agent.
        Top-K ready tasks sorted by active rule; padded with -1.
        """
        obs = np.full((K, N_METRICS), -1.0, dtype=np.float32)
        ready = self._ready_for(agent)
        if ready:
            sorted_ready = self._sort_ready(ready, self._active_rule[agent])
            for slot, task_idx in enumerate(sorted_ready[:K]):
                obs[slot] = self._metrics[task_idx]
        return obs.reshape(-1)  # (104,)
