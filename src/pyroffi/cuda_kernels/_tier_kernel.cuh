/**
 * Scaffolding for writing ONE kernel body that runs at any pyroffi::Tier.
 *
 * Factored out of _ls_ik_cuda_kernel.cu once a second kernel needed the same
 * machinery. Provides the three things every tiered kernel repeats:
 *
 *   PYROFFI_TIER_GROUP_VARS(TIER)   rank / size / leader / group_sync()
 *   pyroffi_tier_from_env()         the PYROFFI_IK_TIER knob
 *   PYROFFI_TIER_DISPATCH(...)      the (tier x bucket) launch switch
 *
 * THE CENTRAL IDEA — write every cooperative loop in GLASS's own form:
 *
 *     for (int i = rank; i < n; i += size) { ... }
 *
 * At Tier::Thread rank/size are (0, 1), so each such loop collapses back to the
 * original sequential code and the thread tier stays byte-for-byte what it was
 * before tiering. The warp and block tiers then spread the same body across 32 or
 * blockDim.x lanes with no separate implementation. This is exactly how GLASS
 * shares one `*_impl` between its own scopes (see its barrier.cuh).
 *
 * BARRIERS: use `group_sync()`, never a raw `__syncthreads()`. At the thread tier
 * the seed guard (`if (s >= n_seeds) return;`) retires lanes RAGGEDLY, so a
 * block-wide barrier there would have divergent participation — UB. group_sync()
 * compiles to nothing at the thread tier, `__syncwarp()` at the warp tier, and
 * `__syncthreads()` at the block tier.
 *
 * SEED INDEXING: the seed a group works on MUST be group-uniform (derive it from
 * the warp id at the warp tier, blockIdx.x at the block tier) so that the seed guard
 * retires a whole group at once. Never derive it from the lane.
 *
 * FK INDEPENDENCE: nothing here reaches inside forward kinematics. The tiers
 * parallelize AROUND the FK — distributing the line search's independent FK calls
 * across lanes — so swapping the FK implementation (e.g. to a 4x4-homogeneous-matrix
 * kernel) does not disturb any of this. Keep it that way: parallelize the callers of
 * FK, not its internals.
 */

#pragma once

#include "_glass_solve.cuh"

#include <cstdlib>
#include <cstring>
#include <type_traits>

/**
 * Declare `rank`, `size`, `leader`, and `group_sync()` for the group owning one seed.
 *
 * rank/size: (0,1) thread | (lane,32) warp | (threadIdx.x, blockDim.x) block.
 * Place at the top of a kernel templated on `pyroffi::Tier TIER`.
 */
#define PYROFFI_TIER_GROUP_VARS(TIER)                                              \
    const int rank = ((TIER) == pyroffi::Tier::Thread) ? 0                         \
                   : ((TIER) == pyroffi::Tier::Warp)   ? (int)(threadIdx.x & 31u)  \
                   :                                     (int)threadIdx.x;         \
    const int size = ((TIER) == pyroffi::Tier::Thread) ? 1                         \
                   : ((TIER) == pyroffi::Tier::Warp)   ? 32                        \
                   :                                     (int)blockDim.x;          \
    const int leader = (rank == 0);                                                \
    auto group_sync = [&]() {                                                      \
        if constexpr ((TIER) == pyroffi::Tier::Warp)  __syncwarp();                \
        if constexpr ((TIER) == pyroffi::Tier::Block) __syncthreads();             \
    };                                                                             \
    (void)leader; (void)group_sync

/**
 * Seed index for this group, and the shared-slot index within the block.
 *
 * Group-uniform by construction — see the SEED INDEXING note above.
 */
#define PYROFFI_TIER_SEED_INDEX(TIER)                                              \
    (((TIER) == pyroffi::Tier::Thread)                                             \
        ? (int)(blockIdx.x * blockDim.x + threadIdx.x)                             \
        : ((TIER) == pyroffi::Tier::Warp)                                          \
            ? (int)(blockIdx.x * (blockDim.x / 32u) + (threadIdx.x / 32u))         \
            : (int)blockIdx.x)

#define PYROFFI_TIER_SLOT(TIER) \
    (((TIER) == pyroffi::Tier::Warp) ? (int)(threadIdx.x / 32u) : 0)

/**
 * Shared slots per block at this tier: one per resident seed.
 * Thread/Block tiers keep a single slot; the warp tier gets one per warp, bounded by
 * the shared A[N*N] each warp owns (see pyroffi::warp_tier_warps_per_block).
 */
#define PYROFFI_TIER_SLOTS(TIER, N) \
    (((TIER) == pyroffi::Tier::Warp) ? pyroffi::warp_tier_warps_per_block((int)(N)) : 1)

/**
 * Shared dimension for a tier's `A` buffer.
 *
 * The thread tier keeps `A` in thread-local storage so nvcc can promote it to
 * registers — the tier's whole premise. A live `__shared__ double A[N*N]` it never
 * reads would be 8KB at N=32 and would cut exactly the occupancy the tier exists for.
 * A `__shared__` declaration cannot be compiled out, so the thread tier's copy is
 * sized down to a scalar stub instead.
 */
#define PYROFFI_TIER_SMEM_N(TIER, N) \
    (((TIER) == pyroffi::Tier::Thread) ? 1 : (int)(N))

/**
 * Parallelism tier from `PYROFFI_IK_TIER` (thread|warp|block); default thread.
 *
 * Read ONCE (function-local static, thread-safe init): this is a benchmarking and
 * autotuning knob, not a per-call parameter, and re-reading the environment per launch
 * would land in the very timings an autotune collects. Unrecognized values fall back to
 * the default rather than failing the launch.
 *
 * Default rationale (measured, RTX A5000 / sm_86): the thread tier wins at large batch
 * (>= ~4k seeds) at every DOF, because it amortizes 32 seeds across a warp while the
 * warp tier's floor is the serial parent->child FK chain. The block tier wins the
 * small-batch corner, where the thread tier is latency-bound and leaves the GPU idle.
 *
 * ONLY CONSULTED AT BUCKETS <= pyroffi::TIER_CHOICE_MAX_N (32). Above that the
 * dispatch forces Tier::Block and ignores this value, because the thread and warp
 * tiers do not fit at N>32 (see tier_choice_allowed's measurements) — so a robot past
 * 32 actuated DOF runs on the block tier whatever PYROFFI_IK_TIER says. The knob stays
 * process-global while the bucket is per-call, so this is resolved per launch, not here.
 */
static pyroffi::Tier pyroffi_tier_from_env()
{
    static const pyroffi::Tier tier = []() {
        const char* e = std::getenv("PYROFFI_IK_TIER");
        if (e) {
            if (std::strcmp(e, "warp")  == 0) return pyroffi::Tier::Warp;
            if (std::strcmp(e, "block") == 0) return pyroffi::Tier::Block;
        }
        return pyroffi::Tier::Thread;
    }();
    return tier;
}

/**
 * Launch KERNEL<TIER, N_> with the per-tier grid/block shape, for ONE bucket.
 *
 * `N_TAG` is a `std::integral_constant<uint32_t, N>` rather than a plain constant on
 * purpose: this body lives inside a generic lambda (see PYROFFI_TIER_DISPATCH), which
 * makes it a TEMPLATE. That is what lets the `if constexpr` below actually discard the
 * thread/warp branches at N>32 — in a non-template function both branches would still
 * be instantiated, and KERNEL<Tier::Warp, 48> is a hard ptxas error ("uses too much
 * shared data"), not a dead branch the compiler can drop.
 *
 * The warp tier's warps-per-block comes from the same constexpr the kernel sizes its
 * shared array with, so host and device cannot disagree about it.
 */
#define PYROFFI_TIER_LAUNCH_BODY_(KERNEL, N_TAG, TIER_VAR, NPROB, NSEEDS,             \
                                  THREAD_TPB, BLOCK_TPB, STREAM, ...)                 \
    do {                                                                              \
        constexpr uint32_t N_ = decltype(N_TAG)::value;                               \
        if constexpr (pyroffi::tier_choice_allowed((int)N_)) {                        \
            switch (TIER_VAR) {                                                       \
            case pyroffi::Tier::Thread: {                                             \
                const int tpb_ = ((NSEEDS) < (THREAD_TPB)) ? (NSEEDS) : (THREAD_TPB); \
                const int bx_  = ((NSEEDS) + tpb_ - 1) / tpb_;                        \
                KERNEL<pyroffi::Tier::Thread, N_>                                     \
                    <<<dim3(bx_, (NPROB)), tpb_, 0, (STREAM)>>>(__VA_ARGS__);         \
                break; }                                                              \
            case pyroffi::Tier::Warp: {                                               \
                constexpr int wpb_ = pyroffi::warp_tier_warps_per_block((int)N_);     \
                const int bx_ = ((NSEEDS) + wpb_ - 1) / wpb_;                         \
                KERNEL<pyroffi::Tier::Warp, N_>                                       \
                    <<<dim3(bx_, (NPROB)), wpb_ * 32, 0, (STREAM)>>>(__VA_ARGS__);    \
                break; }                                                              \
            case pyroffi::Tier::Block:                                                \
                KERNEL<pyroffi::Tier::Block, N_>                                      \
                    <<<dim3((NSEEDS), (NPROB)), (BLOCK_TPB), 0, (STREAM)>>>(          \
                        __VA_ARGS__);                                                 \
                break;                                                                \
            }                                                                         \
        } else {                                                                      \
            /* N > TIER_CHOICE_MAX_N: block tier only — the other two do not fit.    \
               PYROFFI_IK_TIER is IGNORED here rather than honoured-then-failed. */   \
            KERNEL<pyroffi::Tier::Block, N_>                                          \
                <<<dim3((NSEEDS), (NPROB)), (BLOCK_TPB), 0, (STREAM)>>>(__VA_ARGS__); \
        }                                                                             \
    } while (0)

/** One `case` of the bucket switch. Refers only to `pyroffi_launch_` (fixed name), so
 *  it can be expanded by the PYROFFI_SOLVE_N_BUCKETS X-macro, which cannot forward the
 *  dispatch's own arguments. */
#define PYROFFI_TIER_DISPATCH_CASE_(NVAL)                                             \
    case (NVAL):                                                                      \
        pyroffi_launch_(std::integral_constant<uint32_t, (NVAL)>{});                  \
        break;

/**
 * Dispatch KERNEL over (tier x compile-time N bucket).
 *
 * `BUCKET` must come from pyroffi::solve_bucket(n_act); 0 (n_act past the largest
 * bucket) must be rejected by the caller before reaching here.
 *
 * The case list is GENERATED from PYROFFI_SOLVE_N_BUCKETS — the same X-macro
 * solve_bucket() is generated from. This is load-bearing: the two were previously
 * written out by hand and could disagree, and a bucket with no case here does not
 * fail — it falls to `default`, launches NOTHING, and the FFI returns success over an
 * uninitialized output buffer. Keep both sides generated.
 */
#define PYROFFI_TIER_DISPATCH(KERNEL, BUCKET, TIER_VAR, NPROB, NSEEDS,                \
                              THREAD_TPB, BLOCK_TPB, STREAM, ...)                     \
    do {                                                                              \
        auto pyroffi_launch_ = [&](auto n_tag_) {                                     \
            PYROFFI_TIER_LAUNCH_BODY_(KERNEL, n_tag_, TIER_VAR, NPROB, NSEEDS,        \
                                      THREAD_TPB, BLOCK_TPB, STREAM, __VA_ARGS__);    \
        };                                                                            \
        switch (BUCKET) {                                                             \
        PYROFFI_SOLVE_N_BUCKETS(PYROFFI_TIER_DISPATCH_CASE_)                          \
        default:                                                                      \
            /* Unreachable by construction: every bucket solve_bucket() can return    \
               has a case above (same X-macro), and 0 is caller-rejected. */          \
            break;                                                                    \
        }                                                                             \
    } while (0)
