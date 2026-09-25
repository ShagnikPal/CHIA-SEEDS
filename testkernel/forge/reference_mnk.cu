

#include <cuda_runtime.h>

__global__ void sgemm_ref_mnk(const float *A, const float *B, float *C,
                              int M, int N, int K) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;
  float acc = 0.0f;
  for (int i = 0; i < K; ++i) acc += A[row * K + i] * B[i * N + col];
  C[row * N + col] = acc;
}

void launch(const float *A, const float *B, float *C, int M, int N, int K) {
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  sgemm_ref_mnk<<<grid, block>>>(A, B, C, M, N, K);
}
