// Benchmark: pyroffi block_reduce_sum vs glass::reduce (tree) vs glass::reduce_fast.
//
// Apples-to-apples task: every thread in a block holds one register value; produce
// the block-wide sum, available to ALL threads. That is block_reduce_sum's contract,
// so the array-based glass::reduce is charged for the smem[tid]=val staging it needs
// to accept the same input -- that staging is part of the cost of using it here.
#include "glass.cuh"
#include <cstdio>
#include <cmath>

#define N_ITERS 2000

// ── pyroffi's current implementation (copied verbatim from _sco_trajopt) ────
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
__device__ float block_reduce_sum(float val, float* smem, int tid, int bdim) {
    int lane = tid & 31, warp_id = tid >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();
    int n_warps = bdim >> 5;
    float ws = (tid < n_warps) ? smem[tid] : 0.0f;
    ws = warp_reduce_sum(ws);
    if (tid == 0) smem[0] = ws;
    __syncthreads();
    return smem[0];
}

__global__ void k_pyroffi(const float* in, float* out, int bdim) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    float v = in[blockIdx.x * bdim + tid];
    float acc = 0.0f;
    for (int it = 0; it < N_ITERS; it++) {
        acc = block_reduce_sum(v + acc * 1e-30f, smem, tid, bdim);
        __syncthreads();          // caller-owned: smem is reused next iteration
    }
    if (tid == 0) out[blockIdx.x] = acc;
}

// glass::reduce -- array-based tree reduce, destructive. Needs the block's values
// staged into a length-bdim shared array first.
__global__ void k_glass_reduce(const float* in, float* out, int bdim) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    float v = in[blockIdx.x * bdim + tid];
    float acc = 0.0f;
    for (int it = 0; it < N_ITERS; it++) {
        smem[tid] = v + acc * 1e-30f;
        __syncthreads();
        glass::reduce<float>((uint32_t)bdim, smem);
        acc = smem[0];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = acc;
}

// glass::reduce_fast -- register-partial overload: same contract as block_reduce_sum.
__global__ void k_glass_fast(const float* in, float* out, int bdim) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    float v = in[blockIdx.x * bdim + tid];
    float acc = 0.0f;
    for (int it = 0; it < N_ITERS; it++) {
        acc = glass::reduce_fast<float>(v + acc * 1e-30f, smem);
    }
    if (tid == 0) out[blockIdx.x] = acc;
}

// TRAILING_SYNC=false: caller owns the following barrier (what pyroffi's does).
__global__ void k_glass_fast_nots(const float* in, float* out, int bdim) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    float v = in[blockIdx.x * bdim + tid];
    float acc = 0.0f;
    for (int it = 0; it < N_ITERS; it++) {
        acc = glass::reduce_fast<float, false>(v + acc * 1e-30f, smem);
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = acc;
}

#define NBLOCKS 1024

float time_kernel(void(*k)(const float*,float*,int), const float* d_in, float* d_out,
                  int bdim, size_t smem_bytes, float* h_out)
{
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    k<<<NBLOCKS, bdim, smem_bytes>>>(d_in, d_out, bdim);   // warmup
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < 5; r++) k<<<NBLOCKS, bdim, smem_bytes>>>(d_in, d_out, bdim);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / 5.0f;
}

int main() {
    const int bdims[] = {64, 128, 256, 512, 1024};
    float *d_in, *d_out;
    cudaMalloc(&d_in, sizeof(float) * NBLOCKS * 1024);
    cudaMalloc(&d_out, sizeof(float) * NBLOCKS);
    float* h_in = new float[NBLOCKS * 1024];
    for (int i = 0; i < NBLOCKS * 1024; i++) h_in[i] = (float)((i % 17) - 8) * 0.125f;
    cudaMemcpy(d_in, h_in, sizeof(float) * NBLOCKS * 1024, cudaMemcpyHostToDevice);

    printf("Block-sum of one per-thread register value, broadcast to all threads.\n");
    printf("%d blocks x %d reductions, mean of 5 launches. RTX A5000 / sm_86.\n\n", NBLOCKS, N_ITERS);
    printf("%6s | %12s | %12s | %12s | %12s | %s\n", "bdim",
           "pyroffi", "glass::reduce", "reduce_fast", "rf(nots)", "result agreement");
    printf("-------|--------------|--------------|--------------|--------------|------------------\n");

    for (int bi = 0; bi < 5; bi++) {
        int bdim = bdims[bi];
        float r_py, r_gr, r_gf, r_gn;
        // pyroffi + reduce_fast need ceil(bdim/32) floats; glass::reduce needs bdim.
        size_t sm_small = ((bdim + 31) / 32) * sizeof(float);
        size_t sm_full  = bdim * sizeof(float);
        float t_py = time_kernel(k_pyroffi,          d_in, d_out, bdim, sm_small, &r_py);
        float t_gr = time_kernel(k_glass_reduce,     d_in, d_out, bdim, sm_full,  &r_gr);
        float t_gf = time_kernel(k_glass_fast,       d_in, d_out, bdim, sm_small, &r_gf);
        float t_gn = time_kernel(k_glass_fast_nots,  d_in, d_out, bdim, sm_small, &r_gn);
        const char* agree = (r_py == r_gf && r_py == r_gn)
                          ? ((r_py == r_gr) ? "all bit-identical" : "pyroffi==fast; tree differs")
                          : "MISMATCH";
        printf("%6d | %9.3f ms | %9.3f ms | %9.3f ms | %9.3f ms | %s\n",
               bdim, t_py, t_gr, t_gf, t_gn, agree);
    }
    printf("\nsmem: pyroffi & reduce_fast = ceil(bdim/32) floats; glass::reduce = bdim floats.\n");
    delete[] h_in; cudaFree(d_in); cudaFree(d_out);
    return 0;
}
