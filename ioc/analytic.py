"""Baselines that never solve the forward problem.

All three methods here fit theta using only quantities evaluated *at the
demonstration* (or at a denoised version of it).  They cost zero forward
trajectory-optimization solves, which makes them unbeatable on price -- and they
buy that price with the assumption that the demonstration is (near) optimal.
The experiments in this package are largely about where that assumption breaks:
Inverse KKT is exact and free at sigma = 0, and falls below random weights by
sigma = 0.05.  The EIV-TLS method (Rickenbach, Scampicchio & Zeilinger 2024)
addresses this by treating the noisy demonstrations as an errors-in-variables
problem and jointly fitting the denoised trajectory and the cost weights.
"""

import jax
import jax.numpy as jnp

from ioc.outer import adam_scan


def kkt_fit(grad_x, ctxs, demos, K, *, n_steps=400, lr=0.05, return_gram=False,
            basis=None):
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

    G doubles as the **identifiability certificate**: lambda_min ~ 0 means some
    feature combination is unexcited by the demos.  Set `return_gram` to get it.
    When rank(G) < K, refit on the identifiable subspace (see
    `ioc.identifiability`).
    """
    # `basis` rows are the per-feature weight vectors `grad_x` is probed with.
    # The identity is the whitened solver's own basis; a caller that shares ONE
    # unit-scale solver across several fits passes `diag(1/scales)` instead, so
    # the whitening rides in the probe rather than in the solver -- which is
    # what lets those fits be `vmap`ped as one batched program (see
    # `ioc.bench2d.run`).  Both give identical columns.
    e = jnp.eye(K) if basis is None else basis

    def stacked_B(ctx, demo):
        # grad_x is probed once per basis weight; vmapping that axis keeps the
        # graph one call wide instead of K unrolled copies, so compile time
        # stops growing linearly in the number of features.
        x_demo = demo[1:-1].reshape(-1)
        return jax.vmap(grad_x, in_axes=(None, 0, None))(x_demo, e, ctx).T

    Bs = jax.vmap(stacked_B)(ctxs, demos)
    G = jnp.einsum("bik,bil->kl", Bs, Bs) / Bs.shape[0]

    def resid(z):
        theta = jax.nn.softmax(z)
        return theta @ G @ theta

    # The whole fit -- every Adam step -- is one jitted `lax.scan`, so the
    # zero-solve baselines cost one dispatch instead of `n_steps` of them.
    @jax.jit
    def fit(z0):
        return adam_scan(jax.value_and_grad(resid), z0, lr=lr, n_steps=n_steps,
                         return_best=True)[0]

    z = fit(jnp.zeros(K))
    return (z, G) if return_gram else z


def cioc_fit(grad_x, gn_system, ctxs, demos, K, *, n_steps=400, lr=0.05,
             ridge=1e-8, basis=None):
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
    e = jnp.eye(K) if basis is None else basis  # see `kkt_fit` on `basis`

    def per_ctx(ctx, demo):
        # As in `kkt_fit`: the K basis probes are a vmapped axis, not an
        # unrolled Python loop.
        x_demo = demo[1:-1].reshape(-1)
        gs = jax.vmap(grad_x, in_axes=(None, 0, None))(x_demo, e, ctx).T
        Hs = jax.vmap(lambda ek: gn_system(x_demo, ek, ctx)[1])(e)
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

    @jax.jit
    def fit(p0):
        return adam_scan(jax.value_and_grad(obj), p0, lr=lr, n_steps=n_steps,
                         return_best=True)[0]

    params = fit(jnp.zeros(K + 1))
    return params[:-1]


def eiv_fit(grad_x, ctxs, demos, K, *, n_outer=5, n_inner=400, lr=0.05,
            mu=10.0, basis=None):
    """EIV-TLS: IOC as an errors-in-variables problem (Rickenbach et al. 2024).

    Standard Inverse KKT evaluates the stationarity condition B(x~)θ = 0 at the
    noisy demonstration x~.  When x~ = x* + noise, the noise enters the
    regressors B, making the ordinary least-squares estimate biased and
    inconsistent.  The total-least-squares (TLS) fix jointly estimates a
    denoised trajectory U and the cost weights θ:

        min_{U, z}  (1/M) Σ_i ||U_i - x~_i||² / σ²  +  μ · (1/M) Σ_i ||B(U_i) softmax(z)||²

    with the noise covariance σ² re-estimated from the residuals each outer
    iteration (the paper's alternating scheme, Approach 2, Sec. 3.3).

    The penalty μ enforces the KKT stationarity constraint at the *denoised*
    point rather than the noisy demo.  This is the key difference from KKT: the
    fit can move U away from the demo to land on a point that actually satisfies
    stationarity, trading demo fidelity against optimality -- exactly what noise
    demands and what pure KKT cannot do.

    Like KKT and CIOC this is a zero-solve baseline: no forward trajectory
    optimization is needed; the inner problem is replaced by its first-order
    necessary condition.

    The trajectory U is parameterized as a per-context offset δ from the demo's
    interior waypoints, so U_i = x~_i + δ_i.  Only the interior waypoints are
    free (endpoints are boundary conditions, not decision variables).
    """
    e = jnp.eye(K) if basis is None else basis

    M = demos.shape[0]
    # Interior waypoints flattened, same convention as kkt_fit / cioc_fit.
    x_demos = demos[:, 1:-1].reshape(M, -1)  # (M, n_x)
    n_x = x_demos.shape[1]

    def stacked_B(x_flat, ctx):
        return jax.vmap(grad_x, in_axes=(None, 0, None))(x_flat, e, ctx).T

    def objective(params, sigma_sq):
        z = params[:K]
        delta = params[K:].reshape(M, n_x)
        U = x_demos + delta
        theta = jax.nn.softmax(z)

        demo_cost = jnp.mean(jnp.sum(delta ** 2, axis=-1)) / (sigma_sq + 1e-30)

        def kkt_residual(u, ctx):
            B = stacked_B(u, ctx)
            return jnp.sum((B @ theta) ** 2)

        kkt_cost = jnp.mean(jax.vmap(kkt_residual)(U, ctxs))
        return demo_cost + mu * kkt_cost

    @jax.jit
    def fit_one(params0, sigma_sq):
        return adam_scan(
            jax.value_and_grad(lambda p: objective(p, sigma_sq)),
            params0, lr=lr, n_steps=n_inner, return_best=True,
        )[0]

    # Initialize: zero offset, KKT-seeded z.
    z_init = kkt_fit(grad_x, ctxs, demos, K, n_steps=200, lr=lr, basis=basis)
    params = jnp.concatenate([z_init, jnp.zeros(M * n_x)])
    sigma_sq = jnp.array(1.0)

    for _ in range(n_outer):
        params = fit_one(params, sigma_sq)
        delta = params[K:].reshape(M, n_x)
        sigma_sq = jnp.mean(jnp.sum(delta ** 2, axis=-1)) / n_x
        sigma_sq = jnp.maximum(sigma_sq, 1e-12)

    return params[:K]
