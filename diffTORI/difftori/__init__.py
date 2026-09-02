"""DiffTORI: Differentiable Trajectory Optimization for deep RL and imitation
learning (Wan, Wang, Wang, Erickson, Held; NeurIPS 2024, arXiv:2402.05421).

JAX/Flax implementation.  Section and equation numbers in the docstrings refer
to that paper.
"""

from .config import ILConfig, RLConfig, SolverConfig
from .policy_il import DiffTORIPolicy, act, il_loss
from .policy_il import make_solver as make_il_solver
from .agent_rl import (DiffTORIAgent, StateCritic, critic_loss, difftori_loss,
                       plan, planning_horizon, tdmpc_loss)
from .agent_rl import make_solver as make_rl_solver
from .solver import DiffTORISolver, make_difftori_solver
from .data.dataset import ReplayBuffer, SequenceDataset, batches

__all__ = [
    "ILConfig", "RLConfig", "SolverConfig",
    "DiffTORIPolicy", "il_loss", "act", "make_il_solver",
    "DiffTORIAgent", "StateCritic", "plan", "planning_horizon",
    "tdmpc_loss", "difftori_loss", "critic_loss", "make_rl_solver",
    "DiffTORISolver", "make_difftori_solver",
    "ReplayBuffer", "SequenceDataset", "batches",
]
