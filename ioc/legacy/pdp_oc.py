"""A thin, faithful JAX port of Pontryagin Differentiable Programming.

This is a *standalone* re-implementation of Jin et al.'s PDP
(github.com/wanxinjin/Safe-PDP, `SafePDP/PDP.py`), written to mirror the
original method-for-method rather than to reuse anything in `ioc.inner`.  The
point is fidelity: PDP is an *optimal-control* method with an explicit dynamics
model and control variables, and it differentiates the trajectory through the
**auxiliary control system** -- a time-varying LQR obtained by differentiating
the Pontryagin conditions, solved by a Riccati recursion.  This is the canonical
PDP baseline for the package.  Deliberately, it shares nothing with `ioc.inner`:
differentiating that dense weighted-NLLS solve by the implicit function theorem
recovers the same trajectory gradient another way (`implicit_loss_and_grad`
below, kept only as the head-to-head comparator on the same OCP), so building
PDP on it would collapse the two into one method.  Keeping PDP on its own OCP +
Riccati stack is what makes the comparison mean something.

Correspondence to the original `PDP.py`:

    OCSys                        -> `OCSys` (dynamics f, path cost c, final cost h)
    OCSys.ocSolver (IPOPT NLP)   -> `ocsolve` (gradient-based NLP on the controls)
    OCSys.diffPMP / getAuxSys    -> `aux_system` (Hamiltonian derivative matrices)
    LQR.lqrSolver (Riccati)      -> `lqr_aux_solve` (ported line-for-line)
    the IOC imitation gradient   -> `imitation_loss_and_grad`

`implicit_loss_and_grad` is NOT part of PDP; it is the implicit-differentiation
comparator, differentiating the same forward OCP solve via the KKT stationarity
grad_u J = 0, so the two ways of getting d x*/d theta can be compared on one
problem.

Notation follows the paper: the Hamiltonian is H = c + lambda . f, the auxiliary
system's coefficient matrices are (F=f_x, G=f_u, E=f_theta, Hxx, Hxu, Huu, Hxe,
Hue) plus terminal (hxx, hxe), and its "state trajectory" is d x*/d theta.

Everything here is discrete-time, matching the LQR standard form the original
uses internally, X_{k+1} = F_k X_k + G_k U_k + E_k.
"""

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
import optax


@dataclasses.dataclass(frozen=True)
class OCSys:
    """A parameterized discrete-time optimal-control system (the OCSys analogue).

    f(x, u, theta) -> x_next          dynamics          (n,), (m,), (p,) -> (n,)
    c(x, u, theta) -> scalar          path cost
    h(x, theta)    -> scalar          final cost
    n, m, p        state / control / parameter dimensions
    """

    f: Callable
    c: Callable
    h: Callable
    n: int
    m: int
    p: int


# --------------------------------------------------------------------------- #
# Forward optimal-control solve (OCSys.ocSolver).                             #
# The original calls IPOPT on the full NLP; this minimizes the same objective #
# J(u) = sum_t c(x_t,u_t,theta) + h(x_T,theta) over the control sequence with #
# a gradient optimizer, rolling the dynamics out to get the states.  It is a  #
# genuine forward OCP solve and shares no code with `ioc.inner`.             #
# --------------------------------------------------------------------------- #
def rollout(oc: OCSys, x0, us, theta):
    """States from a control sequence: x_{t+1} = f(x_t, u_t, theta)."""

    def step(x, u):
        xn = oc.f(x, u, theta)
        return xn, xn

    _, xs = jax.lax.scan(step, x0, us)
    return jnp.concatenate([x0[None], xs], axis=0)  # (T+1, n)


def total_cost(oc: OCSys, x0, us, theta):
    xs = rollout(oc, x0, us, theta)
    path = jnp.sum(jax.vmap(lambda x, u: oc.c(x, u, theta))(xs[:-1], us))
    return path + oc.h(xs[-1], theta)


def ocsolve(oc: OCSys, x0, theta, T, *, u_init=None, solve_iters=800,
            solve_lr=1e-2):
    """Forward OCP solve: argmin_u J(u).  Returns (x_traj (T+1,n), u_traj (T,m)).

    `solve_iters` / `solve_lr` name the *forward* optimizer so they never
    collide with an outer IOC learning rate threaded through `**solve_kw`.
    """
    us = jnp.zeros((T, oc.m)) if u_init is None else u_init
    opt = optax.adam(solve_lr)
    st = opt.init(us)
    loss_grad = jax.jit(jax.value_and_grad(lambda u: total_cost(oc, x0, u, theta)))

    def body(carry, _):
        us, st = carry
        _, g = loss_grad(us)
        upd, st = opt.update(g, st)
        return (optax.apply_updates(us, upd), st), None

    (us, _), _ = jax.lax.scan(body, (us, st), None, length=solve_iters)
    return rollout(oc, x0, us, theta), us


# --------------------------------------------------------------------------- #
# diffPMP / getAuxSys: the Hamiltonian derivative matrices along x*.          #
# --------------------------------------------------------------------------- #
def _costates(oc: OCSys, xs, us, theta):
    """PMP costates: lambda_T = h_x(x_T); lambda_t = c_x + f_x^T lambda_{t+1}."""
    lam_T = jax.grad(oc.h, argnums=0)(xs[-1], theta)

    def back(lam_next, t):
        x, u = xs[t], us[t]
        cx = jax.grad(oc.c, argnums=0)(x, u, theta)
        fx = jax.jacobian(oc.f, argnums=0)(x, u, theta)
        lam = cx + fx.T @ lam_next
        return lam, lam_next  # emit lambda_{t+1}, the one the Hamiltonian uses

    T = us.shape[0]
    _, lam_next_stack = jax.lax.scan(back, lam_T, jnp.arange(T), reverse=True)
    return lam_next_stack  # (T, n): lam_next_stack[t] = lambda_{t+1}


def aux_system(oc: OCSys, xs, us, theta):
    """Coefficient matrices of the auxiliary control system (getAuxSys).

    Returns time-stacked (F, G, E, Hxx, Hxu, Huu, Hxe, Hue) of length T and the
    terminal (hxx, hxe).  H_t = c(x_t,u_t,theta) + lambda_{t+1} . f(x_t,u_t,theta).
    """
    lam_next = _costates(oc, xs, us, theta)

    def hamil(x, u, th, lam):
        return oc.c(x, u, th) + jnp.dot(lam, oc.f(x, u, th))

    def per_t(x, u, lam):
        F = jax.jacobian(oc.f, argnums=0)(x, u, theta)
        G = jax.jacobian(oc.f, argnums=1)(x, u, theta)
        E = jax.jacobian(oc.f, argnums=2)(x, u, theta)
        Hxx = jax.hessian(hamil, argnums=0)(x, u, theta, lam)
        Hxu = jax.jacobian(jax.grad(hamil, argnums=0), argnums=1)(x, u, theta, lam)
        Huu = jax.hessian(hamil, argnums=1)(x, u, theta, lam)
        Hxe = jax.jacobian(jax.grad(hamil, argnums=0), argnums=2)(x, u, theta, lam)
        Hue = jax.jacobian(jax.grad(hamil, argnums=1), argnums=2)(x, u, theta, lam)
        return F, G, E, Hxx, Hxu, Huu, Hxe, Hue

    mats = jax.vmap(per_t)(xs[:-1], us, lam_next)
    hxx = jax.hessian(oc.h, argnums=0)(xs[-1], theta)
    hxe = jax.jacobian(jax.grad(oc.h, argnums=0), argnums=1)(xs[-1], theta)
    return mats, (hxx, hxe)


# --------------------------------------------------------------------------- #
# LQR.lqrSolver: the Riccati recursion that solves the auxiliary system.      #
# Ported line-for-line from PDP.py (Lemma 4.2).  Its "state trajectory" X_t   #
# is exactly d x*_t / d theta; the batch dimension is the parameter dim p.    #
# --------------------------------------------------------------------------- #
def lqr_aux_solve(mats, terminal, n, m, p):
    """Solve the auxiliary LQR; returns dxdtheta (T+1, n, p) and dudtheta (T, m, p)."""
    F, G, E, Hxx, Hxu, Huu, Hxe, Hue = mats
    hxx, hxe = terminal
    T = F.shape[0]
    I = jnp.eye(n)

    # --- backward Riccati sweep: PP[t], WW[t] (paper's P_t, W_t) ------------- #
    def back(carry, m_t):
        P_next, W_next = carry
        F_t, G_t, E_t, Hxx_t, Hxu_t, Huu_t, Hxe_t, Hue_t = m_t
        invHuu = jnp.linalg.inv(Huu_t)
        GinvHuu = G_t @ invHuu
        HxuinvHuu = Hxu_t @ invHuu
        A_t = F_t - GinvHuu @ Hxu_t.T
        R_t = GinvHuu @ G_t.T
        M_t = E_t - GinvHuu @ Hue_t
        Q_t = Hxx_t - HxuinvHuu @ Hxu_t.T
        N_t = Hxe_t - HxuinvHuu @ Hue_t
        temp = A_t.T @ jnp.linalg.inv(I + P_next @ R_t)
        P_curr = Q_t + temp @ P_next @ A_t
        W_curr = N_t + temp @ (W_next + P_next @ M_t)
        return (P_curr, W_curr), (P_curr, W_curr)

    # Original loops t = T-1 .. 1 producing PP[t-1], with PP[T-1] = hxx.  We
    # scan indices 1..T-1 in reverse; the emitted P_curr/W_curr are PP[0..T-2]
    # in forward order, and PP[T-1] is the terminal seed.
    mats_1 = jax.tree.map(lambda a: a[1:], mats)
    (_, _), (P_head, W_head) = jax.lax.scan(
        back, (hxx, hxe), mats_1, reverse=True)
    PP = jnp.concatenate([P_head, hxx[None]], axis=0)      # (T, n, n)
    WW = jnp.concatenate([W_head, hxe[None]], axis=0)      # (T, n, p)

    # --- forward sweep: X_t = d x_t / d theta ------------------------------- #
    def fwd(X_t, inp):
        F_t, G_t, E_t, Hxu_t, Huu_t, Hue_t, P_next, W_next = inp
        invHuu = jnp.linalg.inv(Huu_t)
        GinvHuu = G_t @ invHuu
        A_t = F_t - GinvHuu @ Hxu_t.T
        M_t = E_t - GinvHuu @ Hue_t
        R_t = GinvHuu @ G_t.T
        U_t = -(invHuu @ (Hxu_t.T @ X_t + Hue_t)) - invHuu @ G_t.T @ (
            jnp.linalg.inv(I + P_next @ R_t)
            @ (P_next @ A_t @ X_t + P_next @ M_t + W_next))
        X_next = F_t @ X_t + G_t @ U_t + E_t
        return X_next, (X_next, U_t)

    X0 = jnp.zeros((n, p))                                 # d x_0 / d theta = 0
    inp = (F, G, E, Hxu, Huu, Hue, PP, WW)
    _, (X_next, U) = jax.lax.scan(fwd, X0, inp)
    dxdtheta = jnp.concatenate([X0[None], X_next], axis=0)  # (T+1, n, p)
    return dxdtheta, U


def traj_and_grad(oc: OCSys, x0, theta, T, **solve_kw):
    """Forward-solve, then get d x*/d theta from the auxiliary system."""
    xs, us = ocsolve(oc, x0, theta, T, **solve_kw)
    mats, terminal = aux_system(oc, xs, us, theta)
    dxdtheta, _ = lqr_aux_solve(mats, terminal, oc.n, oc.m, oc.p)
    return xs, us, dxdtheta


# --------------------------------------------------------------------------- #
# IOC: imitation loss and its PDP (auxiliary-system) gradient.                #
# --------------------------------------------------------------------------- #
def imitation_loss_and_grad(oc: OCSys, x0, theta, demo_xs, T, **solve_kw):
    """L(theta) = sum_t ||x*_t(theta) - demo_t||^2 and dL/dtheta via PDP.

    The gradient is assembled from the auxiliary system's d x*/d theta exactly
    as in the paper -- no autodiff through the forward solve.
    """
    xs, _, dxdtheta = traj_and_grad(oc, x0, theta, T, **solve_kw)
    diff = xs - demo_xs                                    # (T+1, n)
    loss = jnp.sum(diff ** 2)
    grad = 2.0 * jnp.einsum("ti,tip->p", diff, dxdtheta)  # (p,)
    return loss, grad, xs


def implicit_loss_and_grad(oc: OCSys, x0, theta, demo_xs, T, **solve_kw):
    """The IMPLICIT-DIFFERENTIATION comparator -- NOT PDP.

    Same forward OCP solve, but d x*/d theta comes from the implicit function
    theorem on the NLP's stationarity condition grad_u J(u*, theta) = 0:

        d u*/d theta = -(grad^2_uu J)^-1 grad^2_u,theta J,

    a single dense solve with the full control-space Hessian (dimension T*m),
    then d x*/d theta by the chain rule through the rollout.  This is the dense
    KKT / condensed form -- the same quantity PDP's Riccati auxiliary system
    computes, obtained without the temporal LQR structure -- so the two can be
    compared head-to-head on one problem.
    """
    xs, us = ocsolve(oc, x0, theta, T, **solve_kw)
    uflat = us.reshape(-1)

    def J(uf, th):
        return total_cost(oc, x0, uf.reshape(T, oc.m), th)

    Huu = jax.hessian(J, argnums=0)(uflat, theta)                       # (Tm, Tm)
    Hut = jax.jacobian(jax.grad(J, argnums=0), argnums=1)(uflat, theta)  # (Tm, p)
    dudtheta = -jnp.linalg.solve(Huu, Hut)                              # (Tm, p)

    def rf(uf, th):
        return rollout(oc, x0, uf.reshape(T, oc.m), th)                  # (T+1, n)

    dxdu = jax.jacobian(rf, argnums=0)(uflat, theta)                    # (T+1,n,Tm)
    dxdth0 = jax.jacobian(rf, argnums=1)(uflat, theta)                  # (T+1,n,p)
    dxdtheta = jnp.einsum("tiu,up->tip", dxdu, dudtheta) + dxdth0
    diff = xs - demo_xs
    loss = jnp.sum(diff ** 2)
    grad = 2.0 * jnp.einsum("ti,tip->p", diff, dxdtheta)
    return loss, grad, xs


def ioc_fit(oc: OCSys, x0, demo_xs, theta0, T, *, grad_fn=imitation_loss_and_grad,
            lr=1e-2, n_steps=200, **solve_kw):
    """Recover theta by descending the imitation loss L with `grad_fn`.

    `grad_fn` is `imitation_loss_and_grad` (PDP, the default) or
    `implicit_loss_and_grad` (the implicit-diff comparator); both share this
    Adam outer loop so the two methods differ only in how d x*/d theta is
    computed.  Returns (theta, history of (step, loss)).
    """
    opt = optax.adam(lr)
    st = opt.init(theta0)
    theta = theta0
    hist = []
    for i in range(n_steps):
        loss, g, _ = grad_fn(oc, x0, theta, demo_xs, T, **solve_kw)
        upd, st = opt.update(g, st)
        theta = optax.apply_updates(theta, upd)
        hist.append((i, float(loss)))
    return theta, hist


# `pdp_ioc_fit` kept as the PDP-named entry point (thin alias of `ioc_fit`).
def pdp_ioc_fit(oc: OCSys, x0, demo_xs, theta0, T, **kw):
    """First-order PDP recovery (NeurIPS-2020 update): `ioc_fit` with the
    auxiliary-system gradient."""
    return ioc_fit(oc, x0, demo_xs, theta0, T,
                   grad_fn=imitation_loss_and_grad, **kw)


# --------------------------------------------------------------------------- #
# A minimal built-in environment for the standalone baseline / self-test:     #
# a torque-controlled pendulum with a cost linear in three features.          #
# --------------------------------------------------------------------------- #
def pendulum_ocsys(dt=0.05):
    """Discrete pendulum; state=[angle, vel], control=[torque], theta in R^3.

    Cost features: angle^2 (upright regulation), vel^2, torque^2.  Linear in
    theta so IOC recovery of the weights is well posed.
    """

    def f(x, u, theta):
        ang, vel = x[0], x[1]
        acc = jnp.sin(ang) - 0.1 * vel + u[0]  # unit m, g, l; light damping
        return jnp.array([ang + dt * vel, vel + dt * acc])

    def c(x, u, theta):
        phi = jnp.array([x[0] ** 2, x[1] ** 2, u[0] ** 2])
        return jnp.dot(theta, phi)

    def h(x, theta):
        return 10.0 * (x[0] ** 2 + x[1] ** 2)  # fixed terminal, not parameterized

    return OCSys(f=f, c=c, h=h, n=2, m=1, p=3)
