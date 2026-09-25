

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


void launch(const float *A, const float *B, float *C, int M, int N, int K);

#define CK(call)                                                             \
  do {                                                                       \
    cudaError_t _e = (call);                                                 \
    if (_e != cudaSuccess) {                                                 \
      fprintf(stderr, "cuda error %s at %s:%d\n", cudaGetErrorString(_e),    \
              __FILE__, __LINE__);                                           \
      return 2;                                                              \
    }                                                                        \
  } while (0)


static void fill(float *p, long n, unsigned s) {
  unsigned x = s ? s : 1u;
  for (long i = 0; i < n; ++i) {
    x = 1664525u * x + 1013904223u;
    p[i] = (float)((int)((x >> 9) & 0xFFFFu) - 32768) * (1.0f / 32768.0f);
  }
}

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: %s <M> <N> <K> <seed> <dump>\n", argv[0]);
    return 1;
  }
  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int K = atoi(argv[3]);
  const unsigned seed = (unsigned)strtoul(argv[4], nullptr, 10);
  const char *dump = argv[5];
  if (M <= 0 || N <= 0 || K <= 0) { fprintf(stderr, "bad shape\n"); return 1; }

  const size_t eA = (size_t)M * (size_t)K;
  const size_t eB = (size_t)K * (size_t)N;
  const size_t eC = (size_t)M * (size_t)N;
  float *hA = (float *)malloc(eA * sizeof(float));
  float *hB = (float *)malloc(eB * sizeof(float));
  float *hC = (float *)malloc(eC * sizeof(float));
  if (!hA || !hB || !hC) { fprintf(stderr, "host alloc failed\n"); return 2; }
  fill(hA, (long)eA, seed);
  fill(hB, (long)eB, seed ^ 0x9E3779B9u);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CK(cudaMalloc(&dA, eA * sizeof(float)));
  CK(cudaMalloc(&dB, eB * sizeof(float)));
  CK(cudaMalloc(&dC, eC * sizeof(float)));
  CK(cudaMemcpy(dA, hA, eA * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB, eB * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemset(dC, 0, eC * sizeof(float)));

  launch(dA, dB, dC, M, N, K);


  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hC, dC, eC * sizeof(float), cudaMemcpyDeviceToHost));

  FILE *fh = fopen(dump, "wb");
  if (!fh) { fprintf(stderr, "cannot open dump %s\n", dump); return 2; }
  if (fwrite(hC, sizeof(float), eC, fh) != eC) {
    fprintf(stderr, "short write to dump\n");
    fclose(fh);
    return 2;
  }
  fclose(fh);

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  free(hA); free(hB); free(hC);
  return 0;
}
