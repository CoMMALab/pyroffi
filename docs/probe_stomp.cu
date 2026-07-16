// Does stomp's block_reduce_sum_scalar return the block sum to ALL threads?
#include <cstdio>
#include <cmath>
static __device__ __forceinline__ float warp_reduce_sum_scalar(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}
static __device__ __forceinline__ float block_reduce_sum_scalar(float v) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int nwarp = (blockDim.x + 31) >> 5;
    v = warp_reduce_sum_scalar(v);
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();
    float out = (threadIdx.x < nwarp) ? warp_sums[lane] : 0.0f;
    if (wid == 0) out = warp_reduce_sum_scalar(out);
    __syncthreads();
    return out;
}
__global__ void probe(float* out) { out[threadIdx.x] = block_reduce_sum_scalar(1.0f); }
int main() {
    for (int bdim : {8, 32, 64, 128}) {
        float* d; cudaMalloc(&d, sizeof(float)*bdim);
        probe<<<1, bdim>>>(d);
        float h[128]; cudaMemcpy(h, d, sizeof(float)*bdim, cudaMemcpyDeviceToHost);
        int correct = 0; for (int i = 0; i < bdim; i++) if (h[i] == (float)bdim) correct++;
        printf("bdim=%4d  expect %4d on every thread | correct on %3d/%3d threads | "
               "t0=%.0f t1=%.0f t2=%.0f", bdim, bdim, correct, bdim, h[0], h[1], h[2]);
        if (bdim > 32) printf(" t32=%.0f t%d=%.0f", h[32], bdim-1, h[bdim-1]);
        printf("\n");
        cudaFree(d);
    }
    return 0;
}
