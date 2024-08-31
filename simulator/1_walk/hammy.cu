#include <curand_kernel.h>
#include <stdio.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define T 60
#define L 4512
#define MAX_M 5
#define LOOPS 443
#define CHECKPOINTS ((const int[]) {20,30,40,60})
#define CHECKPOINTS_LEN 4

#define __EXTERN extern "C"
#define _32
#define _
#define __WARP_INIT
#define __SYNCTHREADS __syncthreads();
#define __WARP_END

#define TID (threadIdx.x + blockIdx.x * blockDim.x)
#define TID_LOCAL threadIdx.x
__global__ void run(const unsigned long long seed,  unsigned long long* out) {
  __shared__ char checkpoints[L][CHECKPOINTS_LEN];
  __shared__ unsigned long long hist[MAX_M][T / 2 + 1][CHECKPOINTS_LEN];
  __shared__ int counts[T / 2 + 1][CHECKPOINTS_LEN];
  curandStateXORWOW_t state _32;
  unsigned int rnd _32;
  int rnd_left _32 = 0;
  __WARP_INIT
  rnd = 0; // to debug
  curand_init(seed, TID, 0, &state _);
  __WARP_END
  for(int _loop = 0; _loop < LOOPS; ++_loop) {
    __WARP_INIT
    for  (int i = 0; i <= sizeof(counts) / sizeof(**counts) / 32; ++i) {
      if (i * 32 + TID_LOCAL < sizeof(counts) / sizeof(**counts)) {
          ((int *) counts)[i * 32 + TID_LOCAL] = 0;
      }
    }
    for  (int i = 0; i <= sizeof(hist) / sizeof(***hist) / 32; ++i) {
      if (i * 32 + TID_LOCAL < sizeof(hist) / sizeof(***hist)) {
          ((unsigned long long *) hist)[i * 32 + TID_LOCAL] = 0;
      }
    }
    __SYNCTHREADS
    for (int i = 0; i < L / 32; ++i) {
      int checkpoint = 0;
      int current_checkpoint_idx = 0;
      rnd = 0;
      for (int k = 0; k < T; ++k) {
        if (rnd_left _ == 0) {
          rnd = 0;
          rnd _ = curand(&state _);
          rnd_left _ = 32;
        }
        checkpoint += (rnd _ & 1) == 0 ? -1 : 1;
        rnd _ >>= 1;
        --rnd_left _;
        if (k + 1 == CHECKPOINTS[current_checkpoint_idx]) {
          checkpoints[i * 32 + TID_LOCAL][current_checkpoint_idx] = (char) (checkpoint / 2);
          ++current_checkpoint_idx;
        }
      }
    }
    __SYNCTHREADS
    for (int i = 0; i < L / 32; ++i) {
      for (int j = 0; j < CHECKPOINTS_LEN; ++j) {
        if (checkpoints[i * 32 + TID_LOCAL][j] >= -T / 4 && checkpoints[i * 32 + TID_LOCAL][j] <= T / 4) {
            ++counts[checkpoints[i * 32 + TID_LOCAL][j] + T / 4][j];
          }
      }
    }
    __SYNCTHREADS
    int old_count = 0;
    int mode = 0;
    for (int i = 0; i <= 2 * MAX_M; ++i) {
      int idx = (i % 2 == 0 ? 1 : -1) * (MAX_M - i / 2) + T / 4;
      int current_count = counts[idx][CHECKPOINTS_LEN - 1];
      if (current_count > old_count) {
        mode = idx - T / 4;
        old_count = current_count;
      }
    }
    int mode_abs = mode > 0 ? mode : -mode;
    if (mode_abs > 0 && mode_abs <= MAX_M) {
      for (int i = 0; i <= sizeof(*hist) / sizeof(**hist) / 32; ++i) {
        for (int j = 0; j < CHECKPOINTS_LEN; ++j) {
          if (i * 32 + TID_LOCAL < sizeof(*hist) / sizeof(**hist)) {
            hist[mode_abs - 1][i * 32 + TID_LOCAL][j] += mode > 0 ? counts[i * 32 + TID_LOCAL][j] : counts[T / 2 - (i * 32 + TID_LOCAL)][j];
          }
        }
      }
    }
  __WARP_END
  }
  __WARP_INIT
  for (int i = 0; i <= sizeof(hist) / sizeof(***hist) / 32; ++i) {
    if (i * 32 + TID_LOCAL < sizeof(hist) / sizeof(***hist)) {
 //     atomicAdd(&out[i * 32 + TID_LOCAL], ((long long *) hist)[i * 32 + TID_LOCAL]);
    }
  }
  __WARP_END
}

int main() {
    unsigned long long x[10];
    run<<<10000, 32>>>(3232, x);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    return 1;
}