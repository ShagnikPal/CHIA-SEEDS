

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


void launch(const float *A, const float *B, float *C, int N);

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
  if (argc < 4) {
    fprintf(stderr, "usage: %s <N> <seed> <dump>\n", argv[0]);
    return 1;
  }
  const int N = atoi(argv[1]);
  const unsigned seed = (unsigned)strtoul(argv[2], nullptr, 10);
  const char *dump = argv[3];
  if (N <= 0) { fprintf(stderr, "bad N\n"); return 1; }

  const size_t elems = (size_t)N * (size_t)N;
  const size_t sz = elems * sizeof(float);
  float *hA = (float *)malloc(sz), *hB = (float *)malloc(sz);
  float *hC = (float *)malloc(sz);
  if (!hA || !hB || !hC) { fprintf(stderr, "host alloc failed\n"); return 2; }
  fill(hA, (long)elems, seed);
  fill(hB, (long)elems, seed ^ 0x9E3779B9u);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CK(cudaMalloc(&dA, sz));
  CK(cudaMalloc(&dB, sz));
  CK(cudaMalloc(&dC, sz));
  CK(cudaMemcpy(dA, hA, sz, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB, sz, cudaMemcpyHostToDevice));
  CK(cudaMemset(dC, 0, sz));

  launch(dA, dB, dC, N);


  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hC, dC, sz, cudaMemcpyDeviceToHost));

  FILE *fh = fopen(dump, "wb");
  if (!fh) { fprintf(stderr, "cannot open dump %s\n", dump); return 2; }
  if (fwrite(hC, sizeof(float), elems, fh) != elems) {
    fprintf(stderr, "short write to dump\n");
    fclose(fh);
    return 2;
  }
  fclose(fh);

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  free(hA); free(hB); free(hC);
  return 0;
}
