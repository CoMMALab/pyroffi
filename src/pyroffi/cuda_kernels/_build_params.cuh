/**
 * Compile-time capacity limits for the pyroffi CUDA kernels — SINGLE SOURCE OF TRUTH.
 *
 * These are not tuning knobs; they size fixed per-thread arrays (`float delta[MAX_ACT]`,
 * `double A_s[MAX_ACT*MAX_ACT]`, `float T_world[MAX_JOINTS*7]`, …). A robot whose actual
 * DOF exceeds the value the library was COMPILED with will silently run off the end of
 * those arrays — there is no runtime bound in the kernels, and out-of-range local writes
 * are undefined behaviour, not a crash. That is why the limits are (a) asserted here at
 * compile time, and (b) exported below so the Python side can refuse the launch instead
 * of corrupting the stack.
 *
 * Setting them:
 *   bash build_kernels/build_ls_ik_cuda.sh --max-act 24 --max-joints 96
 *   PYROFFI_MAX_ACT=24 bash build_kernels/build_all.sh
 * Defaults live in build_kernels/_build_params.sh and are ALWAYS passed as `-D` by the
 * build scripts; the `#ifndef` fallbacks here only apply to ad-hoc `nvcc` invocations,
 * and are kept numerically identical to that script.
 */

#pragma once

// ── Limits ───────────────────────────────────────────────────────────────────
// Keep these in sync with build_kernels/_build_params.sh (checked by tests).

#ifndef MAX_JOINTS
#define MAX_JOINTS 64
#endif

// 24, not 16: the CRISP-EE benchmarks run panda_allegro (23 actuated DOF), and a
// 16-wide build writes past `delta`/`J`/`A_s` on it (CUDA_ERROR_ILLEGAL_ADDRESS).
// Keeping 24 as the DEFAULT rather than a build flag is deliberate — at 16 a
// plain rebuild silently reverted allegro support.
#ifndef MAX_ACT
#define MAX_ACT 24
#endif

// ── Guardrails ───────────────────────────────────────────────────────────────
// A bad -D lands here as a compile error rather than as stack corruption at runtime.

static_assert(MAX_JOINTS >= 1, "MAX_JOINTS must be >= 1.");
static_assert(MAX_ACT    >= 1, "MAX_ACT must be >= 1.");

// Every actuated DOF is a joint, so the actuated count can never exceed the joint
// count. A build with MAX_ACT > MAX_JOINTS is always a mistake (usually the two -D
// flags swapped) and would silently over-allocate every per-thread array.
static_assert(MAX_ACT <= MAX_JOINTS,
              "MAX_ACT must be <= MAX_JOINTS (every actuated DOF is a joint). "
              "Check the --max-act / --max-joints flags are not swapped.");

// Upper bounds are FOOTPRINT limits, not modelling limits.
//
// MAX_ACT is the binding one. The solve's `double A[N*N]` is the driver, but WHERE it
// lives now depends on the tier (see _glass_solve.cuh): the thread tier keeps it
// per-lane, while the warp/block tiers keep it in shared. So the limit is per-tier,
// and 64 is the ceiling of the only tier that scales.
//
// Measured, `nvcc -Xptxas -v -arch=sm_86`, _ls_ik_cuda_kernel.cu at MAX_JOINTS=64
// (sm_86 static shared limit = 49152 B/block):
//
//   bucket N | Thread: local/thread | Warp: smem/block | Block: smem/block
//        32  |            16,432 B  |      40,480 B    |      15,064 B
//        48  |            26,800 B  |  81,952 FAILS    |      25,432 B
//        64  |            41,264 B  | 139,808 FAILS    |      39,896 B
//
// Block's shared use is ~(6.8KB fixed + 8*N*N), so N=64 sits at 39,896/49,152 (81%)
// and the next bucket up (72) would not fit. Hence 64. Past that needs DYNAMIC shared
// memory (sm_86 opts in to ~99KB via cudaFuncAttributeMaxDynamicSharedMemorySize) and
// a MAX_JOINTS raise to satisfy the MAX_ACT <= MAX_JOINTS assert above — a real
// project, not a constant bump.
//
// Above 32 the dispatch forces the block tier; the thread/warp tiers are not offered
// (_glass_solve.cuh's TIER_CHOICE_MAX_N). Note MAX_ACT sizes per-thread arrays
// (`float J[6*MAX_EE*MAX_ACT]`, cfg, delta, ...) in EVERY bucket's kernel, so a
// MAX_ACT=64 build costs the small arms too — the thread tier's stack frame at bucket
// 32 grows 16,432 -> 18,352 B just by raising MAX_ACT 48 -> 64. Build for the robot
// you deploy; do not raise this "just in case".
static_assert(MAX_ACT <= 64,
              "MAX_ACT > 64: the block tier's shared `double A[N*N]` (~6.8KB + 8*N*N) "
              "exceeds the 48KB static shared budget, and it is the only tier that "
              "scales past 32. Move to dynamic shared memory before raising this.");

// MAX_JOINTS sizes the per-block shared-memory robot model (twists/parent_tf/etc,
// ~O(MAX_JOINTS * 13 floats)) and the per-thread T_world[MAX_JOINTS*7]. 256 joints
// is already ~7KB/thread of pose buffer.
static_assert(MAX_JOINTS <= 256,
              "MAX_JOINTS > 256: per-thread T_world[MAX_JOINTS*7] and the shared "
              "robot model exceed the per-block shared-memory budget.");

// ── Self-report ──────────────────────────────────────────────────────────────
// Exported so Python can read what this .so was ACTUALLY built with (via ctypes)
// and validate a robot against it, rather than assuming the defaults. Without this
// the Python side can only hardcode an assumption — which is exactly how
// _region_ik.py's launch bounds silently decoupled from the build.
//
// Definitions (not just declarations) live in this header because every pyroffi
// .so is compiled from exactly ONE translation unit (see build_kernels/*.sh: each
// script passes a single .cu to nvcc). If that ever changes, these must move into
// a .cu of their own or become `inline`.

extern "C" {

/** Joint capacity this library was compiled with (`-DMAX_JOINTS`). */
__attribute__((visibility("default")))
int pyroffi_max_joints(void) { return MAX_JOINTS; }

/** Actuated-DOF capacity this library was compiled with (`-DMAX_ACT`). */
__attribute__((visibility("default")))
int pyroffi_max_act(void) { return MAX_ACT; }

}  // extern "C"
