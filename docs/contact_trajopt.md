# Contact-Rich, Dynamics-Aware SCO TrajOpt

This document describes pyroffi's contact-rich trajectory optimizer: an
extension of the [SCO TrajOpt](sco_trajopt.md) scaffold that makes the plan
*dynamics-aware* (via GRiD inverse dynamics) and *contact-aware* (via an
augmented-Lagrangian fixed-contact constraint), for bimanual manipulation of a
rigidly-grasped object.

**Code:** [`_contact_trajopt.py`](../src/pyroffi/optimization_engines/_contact_trajopt.py),
[`_contact.py`](../src/pyroffi/dynamics/_contact.py).
**Example:** [`16_00_bimanual_box_lift_contact.py`](../examples/16_00_bimanual_box_lift_contact.py).

---

## 1. Problem

Two arms (each 7-DOF Panda, combined `q = [q_L | q_R] ∈ ℝ¹⁴`) rigidly grasp one
rigid box, pinching it between their grippers on two opposing walls, and lift
it. The box pose is *determined* by the grasp (closed kinematic chain), so it is
not an independent variable.

Per waypoint `t`, the decision variables are:

* `q_t ∈ ℝ¹⁴` — stacked joint configuration of both arms.
* `λ_t ∈ ℝ⁶` — the two world-frame contact forces `[f_L(3) | f_R(3)]`.

We minimize a smooth, low-effort trajectory subject to a fixed-contact
constraint, box dynamics, torque feasibility, and grip validity.

---

## 2. Fixed-contact (grasp closure) constraint

Because the box is rigid, the relative pose of the two grippers must stay
constant along the whole trajectory. Let `T_L(q_t)`, `T_R(q_t)` be the world
gripper poses (per-arm FK composed with the arm's world base transform). The
constant relative transform `c₀ = T_L^{-1} T_R` is captured at the grasp
configuration, and the constraint is the `se(3)` residual

$$g(q_t) = \log\!\big( T_L(q_t)^{-1} T_R(q_t) \; c_0^{-1} \big) \in \mathbb{R}^6,
\qquad g(q_t) = 0.$$

This is the "fixed contact": zero iff the grippers hold the box rigidly.

---

## 3. Box Newton–Euler balance

The box centre is the midpoint of the two world contact points; its world
linear acceleration `a_box(t)` is obtained by central finite differences of the
centre along the trajectory (timestep `dt`). Gravity acts at the centre (no
moment), so the balance is

$$
b(q_t, \lambda_t) =
\begin{bmatrix}
m\,(a_\text{box} - g) - (f_L + f_R) \\[2pt]
(p_L - c)\times f_L + (p_R - c)\times f_R
\end{bmatrix} = 0 ,
$$

a 6-vector (force ; torque about the centre). This couples the contact forces
`λ_t` to the box motion — it is what makes the forces *solved for* rather than
prescribed.

---

## 4. Augmented Lagrangian ("Lagrange operators")

The two equality constraints `g` and `b` are enforced with an **augmented
Lagrangian**. For per-timestep multipliers `μ_t, ν_t` and penalty weights
`ρ_g, ρ_b`, the objective contribution is

$$
\sum_t \Big[ \mu_t^\top g(q_t) + \tfrac{1}{2}\rho_g \|g(q_t)\|^2
           + \nu_t^\top b(q_t,\lambda_t) + \tfrac{1}{2}\rho_b \|b(q_t,\lambda_t)\|^2 \Big].
$$

The multipliers are the "constraint parameters as Lagrange operators": after
each outer iteration they are updated by **dual ascent** at the new iterate,

$$\mu_t \leftarrow \mu_t + \rho_g\, g(q_t), \qquad
  \nu_t \leftarrow \nu_t + \rho_b\, b(q_t,\lambda_t),$$

and the penalties are scaled up (`penalty_scale`, capped). This is the method of
multipliers: it drives the constraints to zero without the ill-conditioning of a
pure quadratic penalty.

---

## 5. Dynamics, torque, and grip terms

* **Effort / torque feasibility.** Per arm, `qd, qdd` come from finite
  differences of the joint trajectory; the arm torque is
  `τ = GRiD.inverse_dynamics(q, qd, qdd, f_ext = −λ)`, where the contact
  reaction enters as a per-body external wrench at the gripper (see
  `arm_contact_fext`). We penalize effort `‖τ‖²` (`w_dynamics`) and a squared
  hinge on `|τ| > τ_max` (`w_torque_limit`). GRiD supplies analytic
  reverse-mode gradients through the inverse-dynamics call.
* **Grip validity.** Each contact force must push into the box
  (`f_n ≥ f_min`) and lie within the Coulomb friction cone
  (`‖f_t‖ ≤ μ f_n`), as squared-hinge penalties.
* **Smoothness / joint limits.** Reused unchanged from
  [SCO TrajOpt](sco_trajopt.md) (`_smoothness_cost`, `_limits_cost`).

---

## 6. Solver structure

```
outer loop (n_outer_iters):
    inner L-BFGS over z = [q | λ]   (n_inner_iters)   # augmented objective
    re-pin q start/goal
    dual ascent:  μ += ρ_g·g,   ν += ρ_b·b
    penalty continuation:  ρ_g, ρ_b *= penalty_scale
```

The inner solver, the L-BFGS two-loop recursion, and the 5-point line search are
imported from [`_sco_optimization.py`](../src/pyroffi/optimization_engines/_sco_optimization.py).
Start and goal `q` are pinned by masking their gradient slices; the contact
forces are free at every waypoint.

---

## 7. GRiD FFI is `vmap`-able (single launch)

The inner L-BFGS line search evaluates the cost at several step sizes with
`jax.vmap`, and the cost calls the GRiD inverse-dynamics FFI kernel. Each GRiD
kernel entry point is wrapped with a true batching rule
(`jax.custom_batching.custom_vmap`, see `_batchable` in
[`_grid_dynamics.py`](../src/pyroffi/dynamics/_grid_dynamics.py)) that folds the
`vmap` axis into the kernel's native leading batch dimension, so a `vmap`
becomes a **single fused kernel launch** — identical (bit-for-bit) to batching
over a leading dim, and differentiable through (forward, `grad`, and
`grad`-of-`vmap`). The `custom_vjp` differentiation layer sits on top and is
`vmap`-transparent, descending into the batchable raw calls on both the forward
and backward pass. One axis is folded per rule application; for many batch axes
at once, prefer a single leading batch dimension over deeply-nested `vmap`.

---

## 8. Configuration reference

| Parameter | Default | Description |
|---|---|---|
| `n_outer_iters` | 12 | Outer AL / penalty-continuation iterations |
| `n_inner_iters` | 40 | L-BFGS steps per outer iteration |
| `m_lbfgs` | 6 | L-BFGS history size |
| `dt` | 0.1 | Waypoint timestep (finite-diff qd/qdd/a_box) |
| `w_smooth`, `w_acc`, `w_jerk` | 1.0, 0.5, 0.1 | Smoothness weights |
| `w_limits` | 1.0 | Joint-limit penalty |
| `w_dynamics` | 1e-3 | Effort `‖τ‖²` weight |
| `w_torque_limit`, `tau_max` | 1.0, 87.0 | Torque-limit hinge weight / limit (N·m) |
| `w_grip`, `mu_friction`, `f_min` | 1.0, 0.6, 2.0 | Grip validity |
| `rho_grasp`, `rho_grasp_max` | 10.0, 1e4 | Grasp-closure AL penalty / cap |
| `rho_box`, `rho_box_max` | 1.0, 1e3 | Box-dynamics AL penalty / cap |
| `penalty_scale` | 2.0 | Per-outer-iter penalty multiplier |
| `dual_scale` | 1.0 | Dual-ascent step scaling |
| `w_force_reg` | 1e-4 | Contact-force regularization |

### Tuning tips

- If the grasp drifts (residual stays high), raise `rho_grasp` /
  `penalty_scale`, or add outer iterations.
- If the solver stalls or forces blow up, lower `penalty_scale` (slower, better
  conditioned) and keep `w_force_reg` non-zero.
- Ensure the **endpoints are grasp-feasible**: pinning an infeasible goal fights
  the fixed-contact constraint. Translating both gripper targets by the same
  vector preserves the grasp offset (see the example's lift construction).

---

## 9. Backends

Only the JAX backend exists today. It runs on the GPU and calls the GRiD
inverse-dynamics FFI kernel (with analytic gradients) inside the inner loop.
`contact_sco_trajopt(..., use_cuda=True)` is reserved for a future full-CUDA
mirror and currently raises `NotImplementedError`; profile the JAX backend
first to decide whether the port is warranted.
