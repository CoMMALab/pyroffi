// Batched TOPP-RA: (1) over boundary-speed pairs (ToppRaPairsFfi), (2) reachability jobs for the
// interval lift (ToppRaReachFfi, below). (1): for every path e and every pair (start speed a, end
// speed b) of its V x V boundary conditions, the time-optimal duration and feasibility.
//
// The inputs are pyroffi.topp's canonical constraints on a uniform arc-length grid (N points,
// spacing delta[e]): rows A [u, x] . <= h at every gridpoint (m rows: acceleration, torque, ...)
// and the velocity bound x <= x_upper, x = sdot^2, u = sddot, x_{i+1} = x_i + 2 delta u_i (the
// discretisation of pyroffi.topp._topp_ra). Per path (one block):
//
//   backward  V passes (one per end speed, a warp each): the controllable sets K_i^b, each from
//             K_{i+1}^b by the 2-variable LP {rows_i, box, K_{i+1}^b reachable} projected onto x --
//             solved exactly by eliminating u (Fourier-Motzkin over the lower / upper u rows)
//   forward   V^2 passes (one per pair, thread a * V + b): TOPP-RA's greedy sweep from x_start^a
//             inside K^b -- at each gridpoint the largest feasible u at the known x_i, no LP
//
// Duration: sum 2 delta / (sdot_i + sdot_{i+1}). A pair is feasible when K^b is nonempty
// everywhere, x_start^a lies in K_0^b, every step has a feasible u, the terminal state admits a u,
// and the schedule never stalls (two consecutive gridpoints at rest). Arithmetic in real.
#include <cfloat>
#include <cmath>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

using real = float;   // the arithmetic type (pyroffi.topp solves in float32 too)

constexpr real kEps = 1e-12f;
constexpr real kRelTol = 1e-4f;      // interval emptiness, relative to the bounds' size (the sets
                                      // are stored in float32, so a state on a set's boundary is
                                      // re-tested with float rounding)
__device__ __forceinline__ bool nonempty(real lo, real hi) {
    return lo <= hi + kRelTol * fmax(1.0f, fabs(lo) + fabs(hi));
}
constexpr real kMinSpeed = 1e-6f;    // pyroffi.topp._topp_ra._MIN_SPEED
constexpr real kXCap = 1e6f, kUCap = 1e6f;

struct Rows {   // gridpoint i's rows, the box and the reachability band
    const float* A;   // [m, 2] (u, x)
    const float* h;   // [m]
    int m;
    // row k normalised to unit length; a u coefficient below kUSnap of it makes the row a pure x
    // bound (a joint that barely moves: dividing by its tiny u coefficient would turn float
    // rounding at the x bound into a huge, contradictory u bound)
    __device__ void row(int k, real* au, real* ax, real* h_) const {
        real a = A[2 * k], b = A[2 * k + 1];
        const real n = sqrt(a * a + b * b);
        if (n < 1e-20f) { *au = 0; *ax = 0; *h_ = h[k]; return; }
        a /= n; b /= n;
        *au = fabs(a) < 1e-7f ? 0.0f : a; *ax = b; *h_ = h[k] / n;
    }
};

// x range of {(u, x): rows, 0 <= x <= x_hi, |u| <= u_cap, r_lo <= x + 2 delta u <= r_hi} (r_*
// unused when delta == 0). Eliminates u: every lower bound on u must sit below every upper bound.
// Warp-collective: the 32 lanes split the rows and (lower, upper) pairs, then reduce.
__device__ bool x_range(const Rows& R, real x_hi, real u_cap, real delta, real r_lo, real r_hi,
                        real* lo_out, real* hi_out) {
    const int lane = threadIdx.x & 31;
    real xlo = 0.0f, xhi = x_hi;
    // the u bounds: index -2 / -1 the band and the box, then the rows; lowers have a_u < 0
    const int base = delta > 0 ? -2 : -1;
    auto bound = [&](int k, bool upper, real* p, real* q) -> bool {   // u <=/>= p + q x
        if (k == -1) { *p = upper ? u_cap : -u_cap; *q = 0.0f; return true; }
        if (k == -2) { *p = (upper ? r_hi : r_lo) / (2 * delta); *q = -1.0f / (2 * delta); return true; }
        real au, ax, h;
        R.row(k, &au, &ax, &h);
        if (upper ? au <= 0.0f : au >= 0.0f) return false;
        *p = h / au; *q = -ax / au;
        return true;
    };
    const int n = R.m - base;   // indices base .. m-1
    for (int k = lane; k < R.m; k += 32) {   // pure x bounds
        real au, ax, h;
        R.row(k, &au, &ax, &h);
        if (au != 0.0f) continue;
        if (fabs(ax) < 1e-20f) { if (h < -1e-9f) { xlo = 1.0f; xhi = 0.0f; } }
        else if (ax > 0) xhi = fmin(xhi, h / ax);
        else xlo = fmax(xlo, h / ax);
    }
    for (int t = lane; t < n * n; t += 32) {   // (lower, upper) pairs: pl + ql x <= pk + qk x
        const int l = t / n + base, k = t % n + base;
        real pl, ql, pk, qk;
        if (!bound(l, false, &pl, &ql) || !bound(k, true, &pk, &qk)) continue;
        const real c = ql - qk, d = pk - pl;
        if (fabs(c) < kEps) { if (d < -1e-9f) { xlo = 1.0f; xhi = 0.0f; } }
        else if (c > 0) xhi = fmin(xhi, d / c);
        else xlo = fmax(xlo, d / c);
    }
    for (int o = 16; o > 0; o >>= 1) {
        xlo = fmax(xlo, __shfl_xor_sync(0xffffffffu, xlo, o));
        xhi = fmin(xhi, __shfl_xor_sync(0xffffffffu, xhi, o));
    }
    *lo_out = xlo; *hi_out = xhi;
    return nonempty(xlo, xhi);
}

// u range at a known x: rows, |u| <= u_cap and (delta > 0) r_lo <= x + 2 delta u <= r_hi.
__device__ bool u_range(const Rows& R, real x, real u_cap, real delta, real r_lo, real r_hi,
                        real* lo_out, real* hi_out) {
    real lo = -u_cap, hi = u_cap;
    if (delta > 0) { lo = fmax(lo, (r_lo - x) / (2 * delta)); hi = fmin(hi, (r_hi - x) / (2 * delta)); }
    for (int k = 0; k < R.m; ++k) {
        real au, ax, h;
        R.row(k, &au, &ax, &h);
        const real rhs = h - ax * x;
        if (au == 0.0f) { if (rhs < -kRelTol * fmax(1.0f, fabs(h) + fabs(ax * x))) return false; continue; }
        if (au > 0) hi = fmin(hi, rhs / au); else lo = fmax(lo, rhs / au);
    }
    *lo_out = lo; *hi_out = fmax(hi, lo);   // within tolerance: the boundary
    return nonempty(lo, hi);
}

__global__ void topp_pairs_kernel(const float* __restrict__ A, const float* __restrict__ H,
                                  const float* __restrict__ xu, const float* __restrict__ delta_in,
                                  const float* __restrict__ xs, const float* __restrict__ xe,
                                  int N, int m, int V, float* __restrict__ k_lo, float* __restrict__ k_hi,
                                  int* __restrict__ back_ok, int* __restrict__ feas, float* __restrict__ dur) {
    const int e = blockIdx.x, t = threadIdx.x;
    const float* Ae = A + (size_t)e * N * m * 2;
    const float* He = H + (size_t)e * N * m;
    const float* XU = xu + (size_t)e * N;
    const real delta = delta_in[e];
    __shared__ real x_cap_sh;
    if (t == 0) {   // pyroffi.topp._topp_ra._solver_scales
        real c = 0.0f;
        for (int i = 0; i < N; ++i) if (isfinite(XU[i])) c = fmax(c, (real)XU[i]);
        x_cap_sh = c > 0 ? c : kXCap;
    }
    __syncthreads();
    const real x_cap = x_cap_sh;
    const real u_cap = delta > 0 ? fmin(fmax(x_cap / (2 * fmax(delta, 1e-12f)), 1e-6f), kUCap) : kUCap;
    auto xhi_at = [&](int i) { return fmin(isfinite(XU[i]) ? (real)XU[i] : kXCap, x_cap); };

    // backward: controllable sets for end speed b, one warp each
    for (int b = t / 32; b < V; b += blockDim.x / 32) {
        const int lane = t & 31;
        float* lo = k_lo + ((size_t)e * V + b) * N;
        float* hi = k_hi + ((size_t)e * V + b) * N;
        const real xend = xe[(size_t)e * V + b];
        real h_next = fmin(xend, xhi_at(N - 1)), l_next = fmin(xend, h_next);
        if (lane == 0) { lo[N - 1] = l_next; hi[N - 1] = h_next; }
        int ok = 1;
        for (int i = N - 2; i >= 0; --i) {
            Rows R{Ae + (size_t)i * m * 2, He + (size_t)i * m, m};
            real l, h;
            const bool f = x_range(R, xhi_at(i), u_cap, delta, l_next, h_next, &l, &h);
            if (!f) { l = 0.0f; h = 0.0f; ok = 0; }
            l = fmin(fmax(l, 0.0f), xhi_at(i)); h = fmin(fmax(h, l), xhi_at(i));
            if (lane == 0) { lo[i] = l; hi[i] = h; }
            l_next = l; h_next = h;
        }
        if (lane == 0) back_ok[(size_t)e * V + b] = ok;
    }
    __syncthreads();
    // forward: greedy from start speed a inside K^b
    for (int p = t; p < V * V; p += blockDim.x) {
        const int a = p / V, b = p % V;
        const float* lo = k_lo + ((size_t)e * V + b) * N;
        const float* hi = k_hi + ((size_t)e * V + b) * N;
        const real xst = xs[(size_t)e * V + a];
        int ok = back_ok[(size_t)e * V + b] && xst <= hi[0] + 1e-6f && xst >= lo[0] - 1e-6f;
        real x = fmin(fmax(xst, (real)lo[0]), (real)hi[0]), T = 0.0f;
        bool stalled = false;
        for (int i = 0; i + 1 < N; ++i) {
            Rows R{Ae + (size_t)i * m * 2, He + (size_t)i * m, m};
            real ul, uh;
            const bool f = u_range(R, x, u_cap, delta, lo[i + 1], hi[i + 1], &ul, &uh);
            const real u = f ? uh : 0.0f;
            ok &= f;
            real xn = fmin(fmax(x + 2 * delta * u, (real)lo[i + 1]), (real)hi[i + 1]);
            xn = fmax(xn, 0.0f);
            const real sum = sqrt(fmax(x, 0.0f) > 1e-12f ? x : 0.0f) + sqrt(xn > 1e-12f ? xn : 0.0f);
            if (sum <= kMinSpeed) stalled = true;
            T += 2 * delta / fmax(sum, kMinSpeed);
            x = xn;
        }
        // the terminal state must admit a control (delta = 0: no successor)
        Rows Rl{Ae + (size_t)(N - 1) * m * 2, He + (size_t)(N - 1) * m, m};
        real ul, uh;
        ok &= u_range(Rl, x, u_cap, 0.0f, 0.0f, x_cap, &ul, &uh);
        const bool degenerate = delta <= 0.0f;
        const size_t o = ((size_t)e * V + a) * V + b;
        feas[o] = degenerate ? 1 : (ok && !stalled);
        dur[o] = degenerate ? 0.f : (float)T;
    }
}

ffi::Error ToppPairsImpl(cudaStream_t stream, ffi::Buffer<ffi::F32> A, ffi::Buffer<ffi::F32> H,
                         ffi::Buffer<ffi::F32> xu, ffi::Buffer<ffi::F32> delta, ffi::Buffer<ffi::F32> xs,
                         ffi::Buffer<ffi::F32> xe, ffi::ResultBuffer<ffi::S32> feas, ffi::ResultBuffer<ffi::F32> dur,
                         ffi::ResultBuffer<ffi::F32> k_lo, ffi::ResultBuffer<ffi::F32> k_hi,
                         ffi::ResultBuffer<ffi::S32> back_ok) {
    const auto d = A.dimensions();   // [E, N, m, 2]
    const int E = d[0], N = d[1], m = d[2], V = xs.dimensions()[1];
    if (E == 0) return ffi::Error::Success();
    const int threads = std::min(1024, std::max(32 * V, ((V * V + 31) / 32) * 32));   // a warp per end speed
    topp_pairs_kernel<<<E, threads, 0, stream>>>(A.typed_data(), H.typed_data(), xu.typed_data(), delta.typed_data(),
                                                  xs.typed_data(), xe.typed_data(), N, m, V, k_lo->typed_data(),
                                                  k_hi->typed_data(), back_ok->typed_data(), feas->typed_data(),
                                                  dur->typed_data());
    const cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? ffi::Error::Success() : ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
}


// Reachability jobs (interval lift): job j runs on path edge[j]; its end state must lie in
// [0, end_hi[j]] (x = sdot^2; 0 = come to rest). One warp sweeps the controllable sets K_i of that
// interval backward (k0[j] = the fastest start x, back_ok[j] = 0 if no x works), then every start
// speed s of the job (threads) runs TOPP-RA's greedy forward sweep inside them: its exact end
// state x_end[j, s], duration and feasibility (a start faster than k0 is infeasible).
__global__ void topp_reach_kernel(const real* __restrict__ A, const real* __restrict__ H, const real* __restrict__ xu,
                                  const real* __restrict__ delta_in, const int* __restrict__ job_edge,
                                  const real* __restrict__ end_hi, const real* __restrict__ xs, int N, int m, int S,
                                  real* __restrict__ k_lo, real* __restrict__ k_hi, real* __restrict__ k0,
                                  int* __restrict__ back_ok, real* __restrict__ x_end, real* __restrict__ dur,
                                  int* __restrict__ feas) {
    const int j = blockIdx.x, t = threadIdx.x, e = job_edge[j];
    const real* Ae = A + (size_t)e * N * m * 2;
    const real* He = H + (size_t)e * N * m;
    const real* XU = xu + (size_t)e * N;
    const real delta = delta_in[e];
    __shared__ real x_cap_sh;
    if (t == 0) {
        real c = 0.0f;
        for (int i = 0; i < N; ++i) if (isfinite(XU[i])) c = fmax(c, (real)XU[i]);
        x_cap_sh = c > 0 ? c : kXCap;
    }
    __syncthreads();
    const real x_cap = x_cap_sh;
    const real u_cap = delta > 0 ? fmin(fmax(x_cap / (2 * fmax(delta, 1e-12f)), 1e-6f), kUCap) : kUCap;
    auto xhi_at = [&](int i) { return fmin(isfinite(XU[i]) ? (real)XU[i] : kXCap, x_cap); };
    real* lo = k_lo + (size_t)j * N;
    real* hi = k_hi + (size_t)j * N;
    if (t < 32) {   // backward, one warp
        real h_next = fmin(fmax(end_hi[j], 0.0f), xhi_at(N - 1)), l_next = 0.0f;
        if (t == 0) { lo[N - 1] = l_next; hi[N - 1] = h_next; }
        int ok = 1;
        for (int i = N - 2; i >= 0; --i) {
            Rows R{Ae + (size_t)i * m * 2, He + (size_t)i * m, m};
            real l, h;
            const bool f = x_range(R, xhi_at(i), u_cap, delta, l_next, h_next, &l, &h);
            if (!f) { l = 0.0f; h = 0.0f; ok = 0; }
            l = fmin(fmax(l, 0.0f), xhi_at(i)); h = fmin(fmax(h, l), xhi_at(i));
            if (t == 0) { lo[i] = l; hi[i] = h; }
            l_next = l; h_next = h;
        }
        if (t == 0) { back_ok[j] = ok; k0[j] = ok ? hi[0] : -1.0f; }
    }
    __syncthreads();
    for (int s = t; s < S; s += blockDim.x) {   // forward, one thread per start speed
        const size_t o = (size_t)j * S + s;
        const real xst = xs[o];
        int ok = back_ok[j] && xst <= hi[0] + kRelTol * fmax(1.0f, hi[0]) && xst >= lo[0] - 1e-6f;
        real x = fmin(fmax(xst, lo[0]), hi[0]), T = 0.0f;
        bool stalled = false;
        for (int i = 0; i + 1 < N; ++i) {
            Rows R{Ae + (size_t)i * m * 2, He + (size_t)i * m, m};
            real ul, uh;
            const bool f = u_range(R, x, u_cap, delta, lo[i + 1], hi[i + 1], &ul, &uh);
            ok &= f;
            real xn = fmax(fmin(fmax(x + 2 * delta * (f ? uh : 0.0f), lo[i + 1]), hi[i + 1]), 0.0f);
            const real sum = sqrt(x > 1e-12f ? x : 0.0f) + sqrt(xn > 1e-12f ? xn : 0.0f);
            if (sum <= kMinSpeed) stalled = true;
            T += 2 * delta / fmax(sum, kMinSpeed);
            x = xn;
        }
        Rows Rl{Ae + (size_t)(N - 1) * m * 2, He + (size_t)(N - 1) * m, m};
        real ul, uh;
        ok &= u_range(Rl, x, u_cap, 0.0f, 0.0f, x_cap, &ul, &uh);
        const bool degenerate = delta <= 0.0f;
        feas[o] = degenerate ? 1 : (ok && !stalled);
        dur[o] = degenerate ? 0.0f : T;
        x_end[o] = degenerate ? xst : x;
    }
}

ffi::Error ToppReachImpl(cudaStream_t stream, ffi::Buffer<ffi::F32> A, ffi::Buffer<ffi::F32> H,
                         ffi::Buffer<ffi::F32> xu, ffi::Buffer<ffi::F32> delta, ffi::Buffer<ffi::S32> job_edge,
                         ffi::Buffer<ffi::F32> end_hi, ffi::Buffer<ffi::F32> xs, ffi::ResultBuffer<ffi::F32> k0,
                         ffi::ResultBuffer<ffi::F32> x_end, ffi::ResultBuffer<ffi::F32> dur,
                         ffi::ResultBuffer<ffi::S32> feas, ffi::ResultBuffer<ffi::F32> k_lo,
                         ffi::ResultBuffer<ffi::F32> k_hi, ffi::ResultBuffer<ffi::S32> back_ok) {
    const auto d = A.dimensions();   // [E, N, m, 2]
    const int N = d[1], m = d[2], J = job_edge.dimensions()[0], S = xs.dimensions()[1];
    if (J == 0) return ffi::Error::Success();
    const int threads = std::max(32, std::min(256, ((S + 31) / 32) * 32));
    topp_reach_kernel<<<J, threads, 0, stream>>>(A.typed_data(), H.typed_data(), xu.typed_data(), delta.typed_data(),
                                                  job_edge.typed_data(), end_hi.typed_data(), xs.typed_data(), N, m, S,
                                                  k_lo->typed_data(), k_hi->typed_data(), k0->typed_data(),
                                                  back_ok->typed_data(), x_end->typed_data(), dur->typed_data(),
                                                  feas->typed_data());
    const cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? ffi::Error::Success() : ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
}
}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ToppRaPairsFfi, ToppPairsImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::S32>>().Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ToppRaReachFfi, ToppReachImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::S32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::S32>>());
