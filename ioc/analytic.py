"""Baselines that never solve the forward problem.

Both methods here fit theta using only quantities evaluated *at the
demonstration*.  They cost zero forward trajectory-optimization solves, which
makes them unbeatable on price -- and both buy that price with the same
assumption, that the demonstration is (near) optimal.  The experiments in this
package are largely about where that assumption breaks: Inverse KKT is exact and
free at sigma = 0, and falls below random weights by sigma = 0.05.
"""

import jax
import jax.numpy as jnp

from ioc.outer import adam


def kkt_fit(grad_x, ctxs, demos, K, *, n_steps=400, lr=0.05, return_gram=False):
    """Inverse KKT / feature matching (Keshavarz et al. 2011; Englert et al. 2017).

    If the demonstration x~ is optimal under theta, it is a stationary point of
    the cost, so the *residual of the KKT conditions* must vanish:

        grad_x J_theta(x~, c) = sum_k theta_k grad_x phi_k(x~, c) = B(c) theta = 0.

    B(c) is the matrix of per-feature gradients at the demonstration and does not
    depend on theta, so the fit is the quadratic program

        min_theta  (1/M) sum_i || B(c_i) theta ||^2  =  theta^T G theta,
        G = (1/M) sum_i B(c_i)^T B(c_i),

    over the simplex.  No forward solve appears anywhere: the inner problem is
    replaced by its first-order optimality condition.  That is both the appeal
    and the failure mode -- a noisy demonstration is not a stationary point of
    *any* cost, so the residual it minimizes is measuring noise, and there is no
    mechanism (unlike a rollout-based loss) that lets the fit trade a small
    stationarity violation for behaviour that matches.

    G doubles as the study's **identifiability certificate**.  Its smallest
    eigenvalue, normalized by its trace, measures how strongly the demonstration
    set excites the weakest direction of the feature basis: lambda_min ~ 0 means
    some combination of features leaves the demonstrations unchanged, so *no*
    method can recover theta along it and a large L1 error there is a property of
    the data, not of the optimizer.  Set `return_gram` to get it.

    When rank(G) = r < K, G is not just a warning -- it is the object the fit
    should be *restricted to*: eigendecompose G = U Lambda U^T, keep the r
    directions above threshold, and refit in the U_r coordinates instead of
    over all K weights.  See `iosp/THEORY_IDENTIFIABLE_REFIT.md` §1-2 (the
    feature-gradient construction here is the `g_ik = grad_x phi_k` Gram of
    that document's §1; the bilevel refit wants the *sensitivity* Gram
    `S_i = dx_i*/dtheta` instead, since that is what the outer loss can see).
    """
    e = jnp.eye(K)

    def stacked_B(ctx, demo):
        x_demo = demo[1:-1].reshape(-1)
        return jnp.stack([grad_x(x_demo, e[k], ctx) for k in range(K)], axis=-1)

    Bs = jax.vmap(stacked_B)(ctxs, demos)
    G = jnp.einsum("bik,bil->kl", Bs, Bs) / Bs.shape[0]

    def resid(z):
        theta = jax.nn.softmax(z)
        return theta @ G @ theta

    obj = jax.jit(jax.value_and_grad(resid))
    z, _ = adam(obj, jnp.zeros(K), lr=lr, n_steps=n_steps)
    return (z, G) if return_gram else z


def cioc_fit(grad_x, gn_system, ctxs, demos, K, *, n_steps=400, lr=0.05, ridge=1e-8):
    """Continuous IOC via the Laplace approximation (Levine & Koltun 2012).

    Under a Boltzmann model p(x) proportional to exp(-J_theta(x)) the partition
    function is intractable.  Expanding J to second order about the demonstration
    x~ and integrating the resulting Gaussian gives

        log p(x~) ~= -1/2 g^T H^-1 g + 1/2 log det H - (d/2) log 2*pi,
        g = grad_x J_theta(x~),   H = grad^2_xx J_theta(x~),

    so the fit minimizes  1/2 g^T H^-1 g - 1/2 log det H  over theta.

    This sits between Inverse KKT and the rollout-based methods: like KKT it
    never solves the forward problem, but rather than merely asserting
    stationarity it *approximates* the trajectory distribution a rollout would
    sample, via a local Gaussian.  The first term is a Hessian-weighted
    stationarity residual (KKT's objective is its unweighted cousin); the log-det
    term is the Gaussian normalizer, and it is what stops the fit from
    degenerating -- it rewards sharply-peaked costs, penalizing the flat
    solutions a pure residual objective is happy with.

    Uses the Gauss-Newton Hessian, which is PSD by construction so log det is
    well defined.
    """
    e = jnp.eye(K)

    def per_ctx(ctx, demo):
        x_demo = demo[1:-1].reshape(-1)
        gs = jnp.stack([grad_x(x_demo, e[k], ctx) for k in range(K)], axis=-1)
        Hs = jnp.stack([gn_system(x_demo, e[k], ctx)[1] for k in range(K)], axis=0)
        return gs, Hs  # (n_x, K), (K, n_x, n_x)

    gs, Hs = jax.vmap(per_ctx)(ctxs, demos)
    eye = jnp.eye(gs.shape[1])

    # Levine & Koltun optimize theta *unconstrained*: their log-det term is
    # exactly what pins down its magnitude (it is the Gaussian normalizer, and it
    # rewards sharply-peaked costs).  Restricting theta to the simplex would fix
    # the scale by fiat while leaving log-det free to distort the direction,
    # which makes the fit far worse than useless (measured: regret 200x random).
    # So carry an explicit log-scale and report the simplex direction.
    def obj(params):
        z, log_s = params[:-1], params[-1]
        th = jnp.exp(log_s) * jax.nn.softmax(z)

        def one(g_k, H_k):
            g = g_k @ th
            H = jnp.einsum("k,kij->ij", th, H_k) + ridge * eye
            _, logdet = jnp.linalg.slogdet(H)
            return 0.5 * jnp.dot(g, jnp.linalg.solve(H, g)) - 0.5 * logdet

        return jnp.mean(jax.vmap(one)(gs, Hs))

    f = jax.jit(jax.value_and_grad(obj))
    params, _ = adam(f, jnp.zeros(K + 1), lr=lr, n_steps=n_steps)
    return params[:-1]
