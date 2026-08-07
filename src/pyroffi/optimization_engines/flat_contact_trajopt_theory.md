# Differential-Flatness Contact-*Aware* Trajopt — A Novice's Guide

This note explains the theory behind
[`_flat_contact_trajopt.py`](_flat_contact_trajopt.py) for someone new to robot
dynamics. It also states honestly what the method does *not* do.

> **Naming.** This solver is **contact-aware**, not contact-rich (see
> [§8](#8-contact-aware-vs-contact-rich)). It reasons about contact forces but
> never *decides* anything about contact — the grasp is one fixed, persistent
> mode, and that fixed mode is *what makes the system flat in the first place*.
> The genuinely contact-rich counterpart, which optimizes the contact forces as
> decision variables and therefore is **not** flat, is
> [`_contact_rich_trajopt.py`](_contact_rich_trajopt.py).

It adapts the core idea of the FLASK paper (Duong et al., *"Ultrafast
Sampling-based Kinodynamic Planning via Differential Flatness"*) to a
manipulation setting. See ["Relation to the paper"](#relation-to-the-paper).

---

## 1. What problem are we solving?

Two (or more) robot arms grab a rigid object — say a box — and carry it from a
start pose to a goal pose. We want a **trajectory**: a sequence of robot
configurations over time that the robot can actually execute while obeying
physics.

## 2. Why is this hard? (Two coupled rules)

**Dynamics, in one sentence:** if you know all the forces on an object, Newton's
laws tell you its acceleration — for translation `F = m·a`, and for rotation
`τ = I·α + ω×(I·ω)`. Trajectory optimization must produce a path whose forces
are physically consistent.

A carry-the-box plan must satisfy two things *at the same time*:

1. **Grasp closure** — every gripper must stay locked to the object. If a hand
   drifts from where it grabbed, the grasp is broken. This *couples the arms*:
   they can't move independently.
2. **Object dynamics** — the finger forces must sum to exactly the force and
   torque needed to accelerate the object along the planned path.

The classic **augmented-Lagrangian** solver treats both as constraints it
*penalizes and fights against*, using extra "dual" bookkeeping variables. It
guesses a path, measures the violation, nudges, and repeats in a slow nested
loop that is easy to get stuck in.

## 3. The key idea: describe the motion by the *object*, not the arms

Instead of tracking *"where are all the joints of every arm?"* (many tangled
numbers), track *"where is the object?"* — just 6 numbers: 3 for position, 3 for
orientation. Call that pose `ξ(t)`. This is the **flat output**.

This is only legal because we **assume the grasp never slips** — the hands are
effectively glued to the object. Under that single assumption:

- **Grasp closure becomes automatic.** Each gripper pose is just
  `ξ · offset_i`, where `offset_i` is the fixed hand-to-object transform
  captured at grab time. Move the object → the hands follow by construction.
  The rule *cannot* be violated because it is baked into the description.
- **Object dynamics becomes automatic.** Once `ξ(t)` is chosen, its
  derivatives give the object's acceleration, so `F = m·a` tells you the exact
  total wrench the fingers must supply. A bit of linear algebra (the **grasp
  map** `G`, and its pseudo-inverse `G⁺`) splits that total among the fingers.
  The forces are *computed*, never guessed — so the dynamics residual is zero
  for *every* candidate path.

Both hard constraints turn into **structure** (true by construction) instead of
**penalties** (fought over). The dual variables and the nested outer loop
disappear, leaving a single, well-conditioned smooth optimization.

## 4. What "differential flatness" actually means

A system is **differentially flat** if there is a small set of outputs (the flat
outputs) such that, once you pick their trajectory, **everything else** — every
joint angle, every contact force — follows from plain formulas and
*derivatives*, with no differential equation left to solve.

Here the object pose `ξ(t)` is that flat output. It's *differential* because the
required forces come from derivatives of the object motion (velocity,
acceleration). Choose the object's dance, and the physics is filled in for free.

## 5. What the solver actually optimizes

The decision vector is:

```
z = [ object-pose deltas (T×6) | joint configs (T×ndof) | squeeze (T) | dt ]
```

- **object-pose deltas** — the flat output `ξ(t)`, as a twist relative to the
  grasp pose.
- **joint configs** — kept so the solver can score torques, joint limits, and
  (future) collisions. These are pulled onto the object-derived gripper poses
  by a soft **tracking cost** (see the catch below).
- **squeeze** — a scalar internal grip force added in the *null space* of the
  grasp map, so it tightens the grip without disturbing the object dynamics.
- **dt** — a shared timestep, so the total duration is itself optimizable
  (enabling a minimum-time objective).

One L-BFGS pass (with a light penalty-continuation on the tracking weight)
solves it — no dual-ascent outer loop.

## 6. The honest catch

- **Flatness here is *manufactured by an assumption*, not discovered.** It holds
  only because we froze the grasp: rigid, persistent, no slip. The moment the
  robot must **let go, regrab, or slide** — a *contact event* — the glue
  assumption breaks, the tidy formulas stop holding, and you are back to a
  genuinely hard, non-smooth problem. Deciding *when and where* to make/break
  contact (the combinatorial core of contact-rich planning) is **out of scope**.
- **The grasp is an input, not an output.** The offsets come from
  `system.grasp_offsets`, captured upstream. This solver assumes a good grasp
  already exists; it does not design one. A bad grasp isn't rejected — it just
  shows up as penalty in the grip-validity/squeeze terms and a worse path.
- **The object is flat; the arms are not.** Choosing `ξ(t)` gives the *object*
  in closed form, but the joint angles that place each hand are recovered by
  optimization (the tracking cost), not by a closed-form inverse.

## 7. Is this a general advance, or a scope narrowing?

Mostly a **narrowing**, with a real but bounded benefit inside it:

- *As a general claim* ("flatness enables contact-rich trajopt") it overreaches:
  this is flatness of a rigid body, valid only within one fixed contact mode.
- *As an engineering choice* ("for rigid-grasp transport, parameterize by object
  pose to kill the coupled constraints and the dual loop") it is a sound,
  reusable improvement: ~4× fewer variables, better conditioning, no outer loop,
  and it applies to any single-object multi-arm transport instance.

It does not expand *which* contact problems are solvable; it makes one
already-solvable-but-ill-conditioned problem much cheaper and cleaner.

## 8. Contact-aware vs. contact-rich

These terms are often used loosely; here is the precise ladder, and where each
solver in this package sits.

| Level | Contact set | Contact forces | Contact schedule | Flat? | Solver |
|---|---|---|---|---|---|
| **Contact-aware** | fixed, given | *allocated* analytically (`G⁺ w_req`) | one persistent mode | **yes** | `flat_contact_trajopt` (this module) |
| **Contact-rich** | fixed, given | **decision variables**, optimized under Newton–Euler | one persistent mode | **no** | `contact_rich_trajopt` |
| **Contact-implicit** | **discovered** | decision variables | **discovered** (complementarity) | no | — (future; intractable fast) |

The distinction that matters:

- **Contact-aware** *uses* contact (it knows the grasp map, friction cones, grip
  validity) but never *decides* anything about it. Because the mode is frozen and
  the forces follow algebraically from the object motion, the problem is flat —
  which is the entire subject of §1–§7. The forces are an *output*.
- **Contact-rich** promotes the contact forces to **first-class decision
  variables**: `z = [q | λ | …]`. It still fixes the contact *mode* (who touches
  what — the same rigid grasp), but it no longer allocates the forces from a
  pseudo-inverse. Instead the object's Newton–Euler balance
  `Σf = m(v̇ − g)`, `Σ(p−c)×f = I ω̇ + ω×(Iω)` becomes an **equality constraint**
  the optimizer must satisfy (here via an augmented Lagrangian), and the friction
  cone shapes *which* balancing forces are admissible. The forces are an
  *unknown*, so there is no flat output and no closed-form substitution — you pay
  with more variables and a dual loop, and you buy back the ability to trade off
  internal forces, respect friction as a live constraint, and (later) generalize
  to non-rigid or multi-point contact where no clean grasp-map inverse exists.
- **Contact-implicit** additionally makes the *schedule* a decision (when to
  make/break contact) via complementarity conditions. This is the combinatorial
  core §6 calls out; it explodes quickly and is deliberately out of scope.

The key structural fact: **flatness and contact-richness are mutually exclusive
here.** The flat solver is flat *because* it declines to optimize the forces; the
moment you make the forces free (contact-rich), the algebraic substitution
`λ = G⁺ w_req` is exactly what you throw away, and flatness goes with it. So
"differential-flatness contact-rich trajopt" was a misnomer — it is
differential-flatness *contact-aware* trajopt, and its contact-rich sibling is a
genuinely different (harder, more general) formulation.

## Appendix: Formal derivation from the manipulator equation

Section 3 asserted flatness intuitively. Here we prove it, starting from the
classic rigid-body manipulator equation of motion with contact:

```
M(q) q̈ + C(q, q̇) q̇ + g(q) = τ + Jc(q)ᵀ λ          (EOM)
```

where
- `q, q̇, q̈` — stacked joint positions/velocities/accelerations of all arms,
- `M(q)` — mass matrix, `C(q,q̇)q̇` — Coriolis/centrifugal, `g(q)` — gravity,
- `τ` — joint torques (the control input),
- `λ = [f_1; …; f_k]` — the `k` contact forces (3 each),
- `Jc(q)` — the **contact Jacobian**, mapping contact forces into joint torques;
  its transpose is how fingertip forces are felt at the joints.

**Goal.** Show there exists a flat output `y` such that `q, q̇, q̈, λ, τ` are all
*algebraic* functions of `y` and finitely many of its time derivatives — the
definition of differential flatness (Def. 1 in the FLASK paper). We will find

```
y = ( ξ , σ )
```

the object pose `ξ ∈ SE(3)` (6 numbers) plus the internal squeeze `σ` (the
grasp-map null-space coordinates). Each piece below shows one block of the
system collapsing onto `y`.

### Step 0 — the two extra physical facts

The bare EOM has more unknowns than equations (`q, λ, τ` all free), so it is
*not* flat on its own. Two facts from the **rigid, persistent grasp** close it:

**(A) Rigid-grasp kinematic constraint.** Each gripper is glued to the object,
so its forward kinematics equals the object pose times a constant offset:

```
FK_i(q_i) = ξ · offset_i         for each arm i          (RG)
```

**(B) Object Newton–Euler.** The grasped body is itself a rigid body whose
motion the contact forces must produce:

```
m_o v̇      = Σ_i f_i + m_o g_vec                          (NE-lin)
I_o ω̇ + ω×(I_o ω) = Σ_i (p_i − c) × f_i                   (NE-ang)
```

with object center `c`, world inertia `I_o = R I_body Rᵀ`, and `v, ω` the
object's linear/angular velocity (the derivatives of `ξ`).

Neither is an extra *modeling approximation*; both are exact consequences of "the
grasp does not slip." They are what turn the underdetermined EOM into a flat
system.

### Step 1 — joint state from the flat output: q, q̇, q̈ = f(ξ, ξ̇, ξ̈)

Differentiate the grasp constraint (RG). The object's spatial velocity `V = (v,ω)`
is `ξ̇` read off in body/world coordinates. The gripper velocity equals the arm
Jacobian times joint rates, and by (RG) it also equals the object twist mapped
through the constant offset (an adjoint `Ad_i`):

```
J_i(q_i) q̇_i = Ad_i · V                                  (RG′)
```

So, arm by arm:

```
q_i  = IK_i(ξ · offset_i)                    ← invert (RG)
q̇_i = J_i(q_i)⁻¹ Ad_i V                      ← solve (RG′)
q̈_i = J_i⁻¹ ( Ad_i V̇ − J̇_i q̇_i )            ← differentiate (RG′)
```

Every joint quantity is now a function of `ξ, ξ̇, ξ̈` alone. **The joint block is
slaved to the object pose.**

> *Non-square caveat.* If an arm is kinematically redundant, `J_i` is not square
> and `IK_i` / `J_i⁻¹` are not unique — there is a self-motion null space. Then
> `q` is *not* uniquely fixed by `ξ`, and strict flatness of the joints fails.
> This is exactly why the implementation keeps `q` as an explicit decision
> variable pinned by a tracking cost rather than substituting a closed-form
> `IK_i`. The *object* stays flat; the redundant *arms* are resolved by the
> optimizer. (See "The object is flat; the arms are not" above.)

### Step 2 — contact forces from the flat output: λ = f(ξ, ξ̇, ξ̈, σ)

Stack the Newton–Euler equations (NE-lin)+(NE-ang) into the **grasp map** `G`:

```
G(ξ, q) λ = w_req ,      w_req = [ m_o(v̇ − g_vec) ;  I_o ω̇ + ω×(I_o ω) ]
```

```
       ⎡  I₃      I₃    …   I₃  ⎤
G  =   ⎣ [p₁−c]×  [p₂−c]× … [p_k−c]× ⎦        (6 × 3k)
```

The right-hand side `w_req` is built entirely from `ξ` and its first two
derivatives (via Step 1 the contact points `p_i(q)` are also functions of `ξ`).
`G` is fat (6 rows, `3k ≥ 6` columns for `k ≥ 2`), so invert it minimum-norm and
add the null-space internal force parameterized by the squeeze `σ`:

```
λ = G⁺ w_req  +  N(G) σ                                   (ALLOC)
```

- `G⁺ w_req` — the unique minimum-norm forces that produce the required object
  wrench (this is [`allocate_forces`](_flat_contact_trajopt.py) line by line);
- `N(G) σ` — internal squeeze that lies in `null(G)`, so it tightens the grip
  **without changing `w_req`**. These null-space coordinates are the *second half
  of the flat output*: they are the degrees of freedom the object pose alone
  cannot pin down. That is why `y = (ξ, σ)`, and why the code carries a `squeeze`
  variable alongside the pose deltas.

So `λ` is an algebraic function of `ξ, ξ̇, ξ̈` and `σ`. **The force block is
determined too — no differential equation solved, just a linear solve.**

### Step 3 — torques from the flat output: τ = f(ξ, ξ̇, ξ̈, σ)

Now every symbol on the right of the EOM is known in terms of the flat output.
Solve the EOM for the control input `τ` (it appears linearly):

```
τ = M(q) q̈ + C(q, q̇) q̇ + g(q) − Jc(q)ᵀ λ
```

Substitute Step 1 for `q, q̇, q̈` and Step 2 for `λ`:

```
τ = τ( ξ, ξ̇, ξ̈, σ )                                      (INV-DYN)
```

This is precisely the inverse-dynamics step. In the implementation it is done
numerically by GRiD's `inverse_dynamics(q, q̇, q̈, f_ext)`, with `f_ext` built
from the allocated `λ` — but symbolically it is nothing more than "plug the
flat-output-derived quantities into the EOM and read off `τ`."

### Conclusion — the flatness map

Collecting the three steps:

```
   flat output              recovered by                         quantity
   y = (ξ, σ)   ──(RG, RG′)──────────────────────────────▶   q, q̇, q̈
                ──(NE + grasp-map pseudo-inverse, ALLOC)──▶   λ
                ──(EOM solved for τ, INV-DYN)────────────▶   τ
```

Every state and input of the coupled arm+object system is an algebraic function
of `y = (ξ, σ)` and its derivatives up to order 2. That is the definition of a
differentially flat system, with `(ξ, σ)` the flat output. Counting checks out:
the flat-output dimension `6 + dim null(G)` equals the number of independent
force/motion degrees the grasp can command — 6 object-wrench directions plus the
internal-force directions — which is the actuation the system genuinely controls.

**Why this is the whole trick.** In a generic trajopt you would carry `q, λ, τ`
as unknowns and enforce (EOM), (RG), (NE) as constraints with duals. The
derivation shows that, *under the rigid-grasp assumption only*, (RG) and (NE) are
not constraints to enforce but **substitutions**: they let you eliminate
`q, λ, τ` and optimize over `(ξ, σ)` directly, with the physics recovered by the
formulas above. Remove the rigid-grasp assumption — allow (RG) to break at a
contact event — and Steps 1–2 lose their closed form, `G` changes rank, and the
substitution collapses back into a genuine constrained problem. That boundary is
the whole story of §6.

## Relation to the paper

FLASK (Duong et al.) uses differential flatness for **free-space, sampling-based
kinodynamic planning**:

| | FLASK paper | this module |
|---|---|---|
| Flat output | the robot's own configuration `q` | the grasped **object's SE(3) pose** `ξ` |
| What flatness buys | closed-form polynomial BVP local paths | grasp closure + object dynamics become structural |
| Solver | sampling-based planner + SIMD collision checking | single L-BFGS trajopt pass |
| Dynamics | robot dynamics, closed form (nilpotent-linear) | contact forces allocated via grasp-map pseudo-inverse |
| Contact | none (collision *avoidance* only) | fixed rigid-grasp mode |

The transplant keeps the paper's *spirit* — "evolve a flat output; recover
everything else structurally" — but changes the flat output (object, not robot),
drops the closed-form nilpotent-BVP machinery, replaces exact arm inversion with
a tracking penalty, and lives entirely inside a single persistent-contact mode.
Notably, the paper itself (§VI remark) flags optimal piecewise-polynomial
motion with **mode switching** (i.e., contact) as hard for high-DOF robots and
leaves it as future work — which is exactly the boundary this module does not
cross.
