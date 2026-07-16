// Verbatim copy of pyroffi's stomp_softmax_kernel + its reduction helpers.
// Softmax invariant: sum_k weights[b,k] == 1.
#include <cstdio>
#include <cmath>
static __device__ __forceinline__ float warp_reduce_sum_scalar(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
static __device__ __forceinline__ float warp_reduce_min_scalar(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fminf(v, __shfl_down_sync(0xffffffff, v, o));
    return v;
}
static __device__ __forceinline__ float block_reduce_sum_scalar(float v) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int nwarp = (blockDim.x + 31) >> 5;
    v = warp_reduce_sum_scalar(v);
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();
    float out = (threadIdx.x < nwarp) ? warp_sums[lane] : 0.0f;
    if (wid == 0) out = warp_reduce_sum_scalar(out);
    __syncthreads();
    return out;
}
static __device__ __forceinline__ float block_reduce_min_scalar(float v) {
    __shared__ float warp_mins[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int nwarp = (blockDim.x + 31) >> 5;
    v = warp_reduce_min_scalar(v);
    if (lane == 0) warp_mins[wid] = v;
    __syncthreads();
    float out = (threadIdx.x < nwarp) ? warp_mins[lane] : 1e30f;
    if (wid == 0) out = warp_reduce_min_scalar(out);
    __syncthreads();
    return out;
}
__global__ void stomp_softmax_kernel(const float* costs, float* weights, int K, float temperature) {
    const int b = blockIdx.x, k = threadIdx.x;
    const float c = (k < K) ? costs[b * K + k] : 1e30f;
    const float min_c = block_reduce_min_scalar(c);
    const float shifted = (k < K) ? (c - min_c) : 0.0f;
    const float sum_shift = block_reduce_sum_scalar((k < K) ? shifted : 0.0f);
    const float mean_shift = sum_shift / (float)K;
    const float diff = shifted - mean_shift;
    const float sum_sq = block_reduce_sum_scalar((k < K) ? (diff * diff) : 0.0f);
    const float std_shift = sqrtf(sum_sq / (float)K);
    const float beta = fmaxf(std_shift, 1e-6f) * temperature;
    const float w = (k < K) ? expf(-shifted / (beta + 1e-18f)) : 0.0f;
    const float sum_w = block_reduce_sum_scalar((k < K) ? w : 0.0f);
    if (k < K) weights[b * K + k] = w / (sum_w + 1e-30f);
}
static int npow2(int x){int p=1;while(p<x)p<<=1;return p;}
int main() {
    for (int K : {8, 16, 32, 64}) {
        float *dc, *dw; cudaMalloc(&dc, sizeof(float)*K); cudaMalloc(&dw, sizeof(float)*K);
        float hc[64]; for (int i = 0; i < K; i++) hc[i] = 1.0f + 0.1f * i;
        cudaMemcpy(dc, hc, sizeof(float)*K, cudaMemcpyHostToDevice);
        stomp_softmax_kernel<<<1, npow2(K)>>>(dc, dw, K, 1.0f);
        float hw[64]; cudaMemcpy(hw, dw, sizeof(float)*K, cudaMemcpyDeviceToHost);
        double s = 0; int nfinite = 0;
        for (int i = 0; i < K; i++) { s += hw[i]; if (isfinite(hw[i])) nfinite++; }
        printf("K=%3d (bdim=%3d): sum(weights)=%-12g  [must be 1.0]   finite=%d/%d   w[0]=%g w[1]=%g\n",
               K, npow2(K), s, nfinite, K, hw[0], hw[1]);
        cudaFree(dc); cudaFree(dw);
    }
    return 0;
}
