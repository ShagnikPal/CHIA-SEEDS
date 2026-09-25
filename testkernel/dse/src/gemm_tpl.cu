

#ifndef TB_M
#define TB_M 128
#endif
#ifndef TB_N
#define TB_N 256
#endif
#ifndef TB_K
#define TB_K 32
#endif
#ifndef WP_M
#define WP_M 64
#endif
#ifndef WP_N
#define WP_N 64
#endif
#ifndef WP_K
#define WP_K 32
#endif
#ifndef STAGES
#define STAGES 3
#endif
#ifndef SPLIT_K
#define SPLIT_K 1
#endif
#ifndef ALIGN_A
#define ALIGN_A 4
#endif
#ifndef ALIGN_B
#define ALIGN_B 4
#endif
#ifndef GEMM_M
#define GEMM_M 1024
#endif
#ifndef GEMM_N
#define GEMM_N 1024
#endif
#ifndef GEMM_K
#define GEMM_K 1024
#endif

#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/device_memory.h>
#include <cstdio>

using Gemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>,
    cutlass::gemm::GemmShape<WP_M, WP_N, WP_K>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    STAGES,
    ALIGN_A, ALIGN_B,
    (SPLIT_K > 1)
>;

int main() {
  int M = GEMM_M, N = GEMM_N, K = GEMM_K;
  cutlass::HostTensor<float, cutlass::layout::RowMajor>    A({M, K});
  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> B({K, N});
  cutlass::HostTensor<float, cutlass::layout::RowMajor>    C({M, N});
  cutlass::HostTensor<float, cutlass::layout::RowMajor>    D({M, N});
  cutlass::reference::host::TensorFillRandomUniform(A.host_view(), 1, 4, -4, 0);
  cutlass::reference::host::TensorFillRandomUniform(B.host_view(), 1, 4, -4, 0);
  cutlass::reference::host::TensorFillRandomUniform(C.host_view(), 1, 4, -4, 0);
  A.sync_device(); B.sync_device(); C.sync_device(); D.sync_device();

  Gemm::Arguments args({M, N, K}, A.device_ref(), B.device_ref(),
                       C.device_ref(), D.device_ref(), {1.0f, 0.0f}, SPLIT_K);
  Gemm op;
  if (op.can_implement(args) != cutlass::Status::kSuccess) {
    printf("CANNOT_IMPLEMENT\n"); return 2;
  }
  size_t ws = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(ws);
  if (op.initialize(args, workspace.get()) != cutlass::Status::kSuccess) {
    printf("INIT_FAIL\n"); return 3;
  }
  if (op() != cutlass::Status::kSuccess) { printf("RUN_FAIL\n"); return 4; }
  cudaDeviceSynchronize();
  D.sync_host();
  printf("OK %dx%dx%d tb=%dx%dx%d wp=%dx%dx%d st=%d sk=%d D[0]=%f\n",
         M, N, K, TB_M, TB_N, TB_K, WP_M, WP_N, WP_K, STAGES, SPLIT_K,
         float(D.host_view().at({0,0})));
  return 0;
}
