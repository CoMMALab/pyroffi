/**
 * Three-tier SPD solve (thread / warp / block) over GLASS, for the pyroffi IK kernels.
 *
 * WHAT THIS REPLACES
 * ------------------
 * `chol_solve(A, b, n)` in _ik_cuda_helpers.cuh: a runtime-sized, single-thread,
 * factor-and-solve that every LM/SQP kernel called from a one-seed-per-thread
 * kernel. It is correct, but it hardcodes ONE parallelism structure — the seed's
 * whole normal-equation solve runs on one thread no matter the DOF. For a 29-DOF
 * G1 that is a 29x29 sequential Cholesky on a single lane while 31 lanes idle.
 *
 * This header instead exposes the SAME solve at three scopes, so a kernel can be
 * instantiated at whichever one wins for a given robot's DOF:
 *
 *   Tier::Thread  1 seed/thread, 32 seeds/warp   glass::thread::posv<T,N>
 *   Tier::Warp    1 seed/warp,   32 lanes/seed   glass::warp::potrf + 2x warp::trsv
 *   Tier::Block   1 seed/block,  all threads     glass::posv<T,N,1,REG,CHECK,REG_DIAG>
 *
 * Which one wins is EMPIRICAL, not a rule — see tools/autotune_backend.py. GLASS's
 * own guidance (N<=7 for the thread tier) is a measured claim about a bare
 * register-resident `T A[N*N]` on nvcc 12.0/sm_86; it is a starting hypothesis
 * here, not a constraint. The autotune measures this repo's kernels on this GPU.
 *
 * WHY COMPILE-TIME `N` (and why padding is safe)
 * ----------------------------------------------
 * GLASS's thread/warp tiers are compile-time-`N` ONLY, on purpose: the tier's
 * value is an `A` that nvcc can keep in registers, which needs fully-unrolled,
 * constant-folded indexing. pyroffi's `n_act` is a RUNTIME kernel argument
 * (`seeds.dimensions()[2]`), so the two cannot meet without a dispatch.
 *
 * We bridge it by padding `n_act` up to a compile-time bucket `N` and templating
 * the kernel on `N`. Padding is numerically inert because it reuses the identity-
 * masking the kernels ALREADY do for fixed joints: a padded index `a` gets a zero
 * row/column, a unit diagonal, and a zero rhs, so the solve returns `x[a] == 0`
 * and the caller's `delta[a]` is 0. See `pad_identity` below.
 *
 * NOTE the bucket boundary is not free: GLASS measured nvcc's local-array
 * promotion threshold between 49 and 64 elements, so a 7-DOF arm padded to N=8
 * (64 doubles) can fall off the register cliff that N=7 (49) sits just inside.
 * That is exactly why 7 is its own bucket rather than folding into 8.
 *
 * LAYOUT
 * ------
 * GLASS is column-major; the pyroffi kernels build `A` row-major. For the normal
 * equations this is a no-op: `A = J^T J + lam*I` is SYMMETRIC, and the fixed-joint
 * masking preserves symmetry (it zeros row AND column), so the row-major buffer
 * and its column-major reading are the same matrix. `posv` returns the solution in
 * `b` and we never read the factor `L` back, so the factor's triangle convention
 * never becomes visible to the caller. Do not reuse this for a NON-symmetric A.
 */

#pragma once

#include "_build_params.cuh"
#include "glass.cuh"

#include <cstdint>

namespace pyroffi {

// ── Tier ─────────────────────────────────────────────────────────────────────
// Scope at which ONE seed's solve is executed. Ordered most->least problem
// packing, matching GLASS's ladder (glass-defaults.cuh `enum class backend`).
enum class Tier { Thread, Warp, Block };

__host__ __device__ constexpr const char* tier_name(Tier t) {
    return t == Tier::Thread ? "thread" : (t == Tier::Warp ? "warp" : "block");
}

/** Threads that cooperate on one seed at this tier (Block resolves at launch). */
__host__ __device__ constexpr int tier_threads_per_seed(Tier t) {
    return t == Tier::Thread ? 1 : (t == Tier::Warp ? 32 : 0);
}

/**
 * Warps per block for the WARP tier at bucket `N` — i.e. seeds resolved per block.
 *
 * Each warp owns a private `double A[N*N]` in shared, so the tier's own footprint
 * bounds its occupancy as DOF grows: at N=32 that is 8KB/warp, and 8 warps/block
 * (65KB) exceeds the 48KB static limit outright (ptxas: "uses too much shared
 * data"). Backing off to 4 warps keeps N=32 under budget.
 *
 * Host and device MUST agree on this: the kernel sizes its shared array from it
 * and the launch computes its grid from it. Hence one constexpr, called by both.
 *
 * Only defined for N <= TIER_CHOICE_MAX_N: past that the warp tier is not offered
 * at all (see tier_choice_allowed), because holding N>32 would force this to 1
 * warp/block — which is just the block tier with 32 threads, not a warp tier.
 */
__host__ __device__ constexpr int warp_tier_warps_per_block(int N) {
    return (N * N * 8 /*bytes*/ * 8 /*warps*/ <= 32768) ? 8 : 4;
}

// ── Which tiers are offered at a given bucket ────────────────────────────────
//
// Above this bucket the tier is FORCED to Block; at or below it, all three tiers
// are offered and the choice is the autotune's (tools/autotune_backend.py).
//
// Not a policy preference — the other two tiers do not fit. Measured with
// `nvcc -Xptxas -v -arch=sm_86` on _ls_ik_cuda_kernel.cu at MAX_JOINTS=64
// (sm_86 static shared limit = 49152 B):
//
//   bucket N | Thread: local/thread | Warp: smem/block | Block: smem/block
//        32  |            16,432 B  |      40,480 B    |      15,064 B
//        48  |            26,800 B  |  81,952 FAILS    |      25,432 B
//        64  |            41,264 B  | 139,808 FAILS    |      39,896 B
//
// Warp is a hard ptxas error past 32 ("uses too much shared data"), not a slowdown.
// Thread compiles but its per-thread `double A[N*N]` reaches 26KB at N=48, which is
// the occupancy collapse this tier exists to avoid — it would be a trap, not a choice.
// Block's shared use is ~(6.8KB fixed + 8*N*N) and is the only one that scales.
constexpr int TIER_CHOICE_MAX_N = 32;

/** True if the caller may pick a tier at bucket `N`; false => Block is forced. */
__host__ __device__ constexpr bool tier_choice_allowed(int N) {
    return N <= TIER_CHOICE_MAX_N;
}

// ── Compile-time N buckets ───────────────────────────────────────────────────
// The instantiated sizes, ASCENDING (solve_bucket relies on the order). 7 and 8 are
// separate ON PURPOSE (see the promotion-threshold note above); the rest cover the
// larger arms. 64 is the ceiling _build_params.cuh's static_assert enforces on
// MAX_ACT, and is where the block tier's shared budget runs out (see
// TIER_CHOICE_MAX_N's table).
//
// Adding a bucket costs one kernel instantiation per OFFERED tier — keep the list
// short and justified by a real robot, not by filling in the sequence.
//
//   7  -> Panda (7 DOF), exact; 49 doubles, inside nvcc's promotion threshold
//   8  -> Fetch (8 DOF), exact; 64 doubles, just past it
//   16 -> Baxter (~14 DOF), padded
//   32 -> padded; the largest bucket offering a tier CHOICE
//   48 -> G1 Unitree with hands (43 DOF), padded; block tier only
//   64 -> padded; block tier only. The last bucket that fits static shared.
//
// NOTE 48/64 are block-tier-only, so they cost ONE instantiation each, not three.
#define PYROFFI_SOLVE_N_BUCKETS(X) X(7) X(8) X(16) X(32) X(48) X(64)

/**
 * Smallest instantiated bucket that holds `n_act`, or 0 if none does.
 *
 * Generated from PYROFFI_SOLVE_N_BUCKETS so it cannot disagree with the set of
 * buckets the dispatch actually instantiates — returning a bucket nobody compiled
 * launches no kernel at all (see PYROFFI_TIER_DISPATCH).
 */
__host__ __device__ constexpr int solve_bucket(int n_act) {
#define PYROFFI_SOLVE_BUCKET_PICK_(NVAL) if (n_act <= (NVAL)) return (NVAL);
    PYROFFI_SOLVE_N_BUCKETS(PYROFFI_SOLVE_BUCKET_PICK_)
#undef PYROFFI_SOLVE_BUCKET_PICK_
    return 0;  // caller must reject: beyond MAX_ACT's ceiling anyway
}

/** Largest supported actuated DOF — the last bucket. Mirrors MAX_ACT's ceiling.
 *  The macro form exists so the FFI handlers can stringify it into their error
 *  literals (PYROFFI_SOLVE_MAX_N_STR) instead of retyping the number. */
#define PYROFFI_SOLVE_MAX_N 64
#define PYROFFI_STRINGIFY_(x) #x
#define PYROFFI_STRINGIFY(x) PYROFFI_STRINGIFY_(x)
#define PYROFFI_SOLVE_MAX_N_STR PYROFFI_STRINGIFY(PYROFFI_SOLVE_MAX_N)
constexpr int SOLVE_MAX_N = PYROFFI_SOLVE_MAX_N;
static_assert(solve_bucket(SOLVE_MAX_N) == SOLVE_MAX_N,
              "SOLVE_MAX_N must be the largest PYROFFI_SOLVE_N_BUCKETS entry.");
static_assert(solve_bucket(SOLVE_MAX_N + 1) == 0,
              "SOLVE_MAX_N must be the largest PYROFFI_SOLVE_N_BUCKETS entry.");
static_assert(MAX_ACT <= SOLVE_MAX_N,
              "MAX_ACT exceeds the largest solve bucket: a robot at MAX_ACT DOF would "
              "get solve_bucket()==0 and no kernel would launch. Add a bucket first.");

// ── Identity padding ─────────────────────────────────────────────────────────

/**
 * Identity-pad the tail of an `n_act x n_act` system already laid out at stride `N`.
 *
 * Writes a zero row/column with a unit diagonal and a zero rhs for every index in
 * `[n_act, N)`, so the solve yields `x[a] == 0` there and the padded DOF cannot
 * perturb the real ones. This is the SAME construction the kernels already use to
 * freeze a fixed joint, so it introduces no new numerical behaviour.
 *
 * PRECONDITION: the caller assembled `A` at stride `N` (i.e. `A[i*N + j]`), NOT at
 * stride `n_act`. Assembling directly at the bucket stride costs nothing — the
 * assembly loop is writing each entry once either way — and avoids an O(N^2) serial
 * repack on the leader, which at N=32 would be 1024 elements moved by one lane while
 * 31 idle.
 *
 * Written in GLASS's `for (i = rank; i < n; i += size)` form so ONE body serves every
 * tier: the thread tier passes (0, 1) and it degenerates to the sequential loop.
 * The caller owns the barrier afterwards.
 *
 * @tparam N      Bucket dimension (compile-time).
 * @param rank    Caller's index within its group (0 for the thread tier).
 * @param size    Group width (1 for the thread tier).
 * @param n_act   True actuated DOF (<= N).
 * @param A       In/out: `N x N` at stride `N`; rows/cols `[n_act, N)` set to identity.
 * @param b       In/out: length `N`; entries `[n_act, N)` zeroed.
 */
template <typename T, uint32_t N>
__device__ __forceinline__ void pad_tail_identity(int rank, int size, int n_act,
                                                  T* __restrict__ A, T* __restrict__ b)
{
    for (int a = n_act + rank; a < (int)N; a += size) {
        for (int j = 0; j < (int)N; j++) A[a*(int)N + j] = A[j*(int)N + a] = static_cast<T>(0);
        A[a*(int)N + a] = static_cast<T>(1);
        b[a] = static_cast<T>(0);
    }
}

// ── The three-tier solve ─────────────────────────────────────────────────────

/**
 * Solve the SPD system `A x = b` at the given tier. `A` is `N x N` (already
 * identity-padded by `pad_identity`); on return `b` holds `x` and `A` holds its
 * Cholesky factor. Mirrors `chol_solve`'s contract, including its non-PD
 * behaviour: on a non-positive-definite pivot the solution is zeroed and `false`
 * is returned, so an LM caller can escalate `lambda` and retry.
 *
 * `A`/`b` residency by tier — the caller owns this and it is NOT interchangeable:
 *   Thread  thread-local (register/local) arrays, one per lane
 *   Warp    SHARED memory, one buffer per warp (all 32 lanes read/write it)
 *   Block   SHARED memory, one buffer per block
 *
 * @tparam TIER  Scope executing the solve.
 * @tparam N     Compile-time bucket dimension.
 * @return false if `A` was not positive-definite (b zeroed), true otherwise.
 */
template <Tier TIER, typename T, uint32_t N>
__device__ __forceinline__ bool tier_posv(T* __restrict__ A, T* __restrict__ b, int* s_fail)
{
    // Only the block tier scales past TIER_CHOICE_MAX_N (see its table). Catch a bad
    // instantiation here, as one readable line, rather than as a ptxas "uses too much
    // shared data" against a mangled name (warp) or a silent 26KB/thread stack frame
    // (thread). PYROFFI_TIER_DISPATCH will not produce these; a hand-written launch might.
    static_assert(TIER == Tier::Block || tier_choice_allowed((int)N),
                  "Only Tier::Block is supported above TIER_CHOICE_MAX_N (=32): the warp "
                  "tier's per-warp shared double A[N*N] overflows the 48KB block budget, "
                  "and the thread tier's per-lane copy collapses occupancy. Use Tier::Block.");

    if constexpr (TIER == Tier::Thread) {
        // No flagged thread::posv exists in GLASS, so the non-PD check is done by
        // factoring with thread::potrf<CHECK=true> and then running the two
        // substitutions ourselves via thread::potrs — same composition
        // thread::posv performs internally, just with the check turned on.
        int fail = 0;
        glass::thread::potrf<T, N, /*CHECK=*/true>(A, &fail);
        if (fail) { for (uint32_t i = 0; i < N; i++) b[i] = static_cast<T>(0); return false; }
        glass::thread::potrs<T, N>(A, b);
        return true;
    } else if constexpr (TIER == Tier::Warp) {
        // GLASS has no warp::posv — compose it from warp::potrf + the two warp::trsv
        // legs (forward `L y = b`, back `L^T x = y`), which is what posv_impl does
        // at block scope. `s_fail` must be a per-warp shared slot: every lane reads it.
        const uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x) & 31u;
        if (lane == 0) *s_fail = 0;
        __syncwarp();
        glass::warp::potrf<T, N, /*CHECK=*/true>(A, s_fail);
        __syncwarp();
        if (*s_fail) {
            for (uint32_t i = lane; i < N; i += 32u) b[i] = static_cast<T>(0);
            __syncwarp();
            return false;
        }
        glass::warp::trsv<T, N, glass::FillMode::Lower, glass::Diag::NonUnit, /*TRANSPOSE=*/false>(A, b);
        glass::warp::trsv<T, N, glass::FillMode::Lower, glass::Diag::NonUnit, /*TRANSPOSE=*/true >(A, b);
        __syncwarp();
        return true;
    } else {
        // Block tier gets GLASS's flagged posv, the only surface with CHECK built in.
        // NRHS=1: the flags live on the multi-RHS overload (a flagged single-RHS form
        // would be ambiguous with it), so a single-RHS flagged solve IS NRHS=1.
        glass::posv<T, N, /*NRHS=*/1, /*REGULARIZE=*/false, /*CHECK=*/true>(A, b, T(0), s_fail);
        __syncthreads();
        if (*s_fail) {
            const uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x;
            const uint32_t size = blockDim.x * blockDim.y;
            for (uint32_t i = rank; i < N; i += size) b[i] = static_cast<T>(0);
            __syncthreads();
            return false;
        }
        return true;
    }
}

}  // namespace pyroffi
