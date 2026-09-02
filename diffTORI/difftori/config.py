"""Hyperparameters for DiffTORI (Wan, Wang, Wang et al., NeurIPS 2024).

Values come from the **released code** (github.com/wkwan7/DiffTORI) wherever it
and the paper disagree, because that is what produced the reported numbers:

  IL   ``DiffTORI_IL_Metaworld/DiffTORI/diffusion_policy_3d/config/difftori.yaml``
       and ``.../config/task/metaworld_*_pointcloud.yaml``
  RL   ``mbrl/cfgs/default.yaml``

Table 9 of the paper is noted where it differs.  Anything neither source pins
down is marked ``# not in either source``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SolverConfig:
    """Inner solver: ``pyroffi.optimization_engines.dynamics_trajopt``.

    The released code runs Theseus' Levenberg--Marquardt; we use this repo's
    dynamics-aware L-BFGS engine, the same one ``ioc.inner`` optimizes through.
    Their ``traj_opt_step`` / ``damping`` have no analogue -- the engine
    line-searches -- so only the iteration budget carries over.
    """

    n_iters: int = 100              # IL traj_opt_num=100 (RL: max_iterations=50)
    grad_tol: float = 1e-6          # engine default; early-stop threshold
    m_lbfgs: int = 8                # engine default
    smooth: bool = True             # soft line search + soft curvature gate
    adjoint_ridge: float = 1e-9     # conditioning only -- see solver.py
    unroll_tail: int = 5            # their backward_num_iterations=5
    action_penalty: float = 1.0     # smooth stand-in for their action clamp


@dataclass(frozen=True)
class ILConfig:
    """DiffTORI for imitation learning, as released.

    Note ``horizon`` is an **action-chunk length**, not a planning horizon over
    a dynamics model: the released policy has no latent dynamics and scores the
    whole chunk with one network.  See ``policy_il``.
    """

    action_dim: int = 4             # MetaWorld; Panda reach-obstacle uses 7
    obs_dim: int = 9                # agent_pos; Panda reach-obstacle uses 25
    horizon: int = 4                # difftori.yaml horizon (action chunk)
    n_obs_steps: int = 2            # difftori.yaml n_obs_steps
    obs_feature_dim: int = 64       # difftori.yaml encoder_output_dim
    posterior_dim: int = 64         # difftori.py z_dim
    mlp_hidden: int = 256           # difftori.yaml mlp_hidden_dim
    action_loss_weight: float = 3000.0   # difftori.yaml action_loss_weight
    kl_coefficient: float = 10.0    # difftori.py _compute_cvae_loss (paper: 1)
    learning_rate: float = 1e-4     # difftori.py Adam lr (paper Table 9: 3e-4)
    lr_min: float = 1e-6            # CosineAnnealingLR eta_min
    lr_schedule_steps: int = 15_000  # CosineAnnealingLR T_max
    batch_size: int = 128           # difftori.py train_batch_size
    grad_norm: float = 0.0          # released code has clipping commented out
    encoder_hidden: int = 256       # not in either source (MLP obs encoder only)
    init_noise: float = 0.0         # difftori.yaml expert_noise
    zero_init: bool = True          # difftori.yaml use_zero_initial=False; see
                                    # policy_il -- True unless a base policy is
                                    # supplied

    # -- Eq. 7 / Appendix D: what the released code drops -------------------
    # The released policy has no latent dynamics and no discounted sum: one
    # network scores a flat action chunk.  Eq. 7 and Appendix D both call for a
    # rollout of ``d_theta`` inside the objective.  These three fields select
    # between the two, defaulting to released-code behaviour so existing runs
    # and checkpoints keep their meaning; ``ILConfig.paper()`` flips them.
    use_dynamics: bool = False      # roll d_theta into the objective (Eq. 7)
    planning_horizon: int = 1       # H in Eq. 7.  Table 9: "Planning horizon
                                    # schedule: 1" -- so the chunk is H+1 = 2
                                    # actions and d_theta is applied once.  NOT
                                    # the same quantity as ``horizon`` above,
                                    # which is the released code's chunk length.
    discount: float = 1.0           # gamma in Eq. 7; 0.99 under paper()
    paper_cvae: bool = False        # Appendix D: encode a* through h^a, and
                                    # order the latent z = [z~, z^s].  The
                                    # released code has no h^a and uses
                                    # [z^s, z~]; the order matters only in that
                                    # it must match between train and test.

    solver: SolverConfig = field(default_factory=SolverConfig)

    @property
    def chunk_len(self) -> int:
        """Number of action steps the decoder optimises.

        Eq. 7 sums ``l = t .. t+H``, so a horizon of ``H`` decides ``H + 1``
        actions.  Released-code mode ignores ``planning_horizon`` and optimises
        a flat ``horizon``-step chunk.
        """
        return self.planning_horizon + 1 if self.use_dynamics else self.horizon

    @classmethod
    def paper(cls, **overrides) -> "ILConfig":
        """Table 9 (Imitation Learning) + Eq. 7 + Appendix D.

        Differs from the released-code defaults above in every value the paper
        actually pins down::

            KL coefficient      1      (released: 10)
            learning rate       3e-4   (released: 1e-4, cosine-annealed)
            latent dimension    50     (released: 64)
            posterior dim       64     (released: 64 -- agrees)
            max planning iters  100    (agrees)
            planning horizon    1      (released: unused; chunk of 4 instead)
            reconstruction wt   1      (released: 3000)

        The reconstruction weight is 1 because Eq. 9 is a plain ELBO: squared
        error minus ``beta * KL``, with no separate action-loss scaling.  The
        released code's 3000/10 pair is a re-tuning of the same two terms.
        """
        base = dict(
            obs_feature_dim=50,      # Table 9 "Latent dimension"
            posterior_dim=64,        # Table 9 "Posterior Gaussian dimension"
            kl_coefficient=1.0,      # Table 9 "KL coefficient"
            learning_rate=3e-4,      # Table 9 "Learning rate"
            action_loss_weight=1.0,  # Eq. 9 is an unweighted ELBO
            use_dynamics=True,
            planning_horizon=1,      # Table 9 "Planning horizon schedule"
            discount=0.99,
            paper_cvae=True,
            solver=SolverConfig(n_iters=100),   # Table 9 "Max planning iterations"
        )
        base.update(overrides)
        return cls(**base)


@dataclass(frozen=True)
class RLConfig:
    """DiffTORI for model-based RL on top of TD-MPC (``mbrl/cfgs/default.yaml``)."""

    action_dim: int = 6
    obs_dim: int = 24               # state dim S for the state-space twin Q
    latent_dim: int = 50
    horizon_start: int = 1          # horizon_schedule: linear(1, 5, 25000)
    horizon_end: int = 5
    horizon_anneal_steps: int = 25_000
    discount: float = 0.99
    action_loss_coefficient: float = 1.0   # c0 in Eq. 6
    rho: float = 0.5                # TD-MPC default
    consistency_coef: float = 2.0   # c3, TD-MPC default
    reward_coef: float = 0.5        # c1, TD-MPC default
    value_coef: float = 0.1         # c2, TD-MPC default
    learning_rate: float = 1e-3
    batch_size: int = 512
    grad_norm: float = 10.0
    mlp_hidden: int = 512
    enc_hidden: int = 256
    tau: float = 0.01
    solver: SolverConfig = field(default_factory=lambda: SolverConfig(n_iters=50))
