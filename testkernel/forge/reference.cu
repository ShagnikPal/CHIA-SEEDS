

#include <cuda_runtime.h>

__global__ void sgemm_ref(const float *A, const float *B, float *C, int N) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < N && c < N) {
    float s = 0.f;
    for (int k = 0; k < N; ++k) s += A[r * N + k] * B[k * N + c];
    C[r * N + c] = s;
  }
}

void launch(const float *A, const float *B, float *C, int N) {
  dim3 t(16, 16), g((N + 15) / 16, (N + 15) / 16);
  sgemm_ref<<<g, t>>>(A, B, C, N);
}
