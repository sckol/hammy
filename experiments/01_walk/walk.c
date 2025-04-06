#ifndef FROM_PYTHON
#include "walk.h"
#endif

STATIC_ASSERT(BINS_LEN == BINS_MAX - BINS_MIN + 1, bins_len_does_not_match);
__device__ unsigned long long counts [BLOCKS][32][TARGETS_LEN][CHECKPOINTS_LEN][BINS_LEN];
__EXTERN __global__ EXPORT void run_simulation(unsigned long long loops, const unsigned long long seed,  unsigned long long* out) {
  __shared__ int positions[CHECKPOINTS_LEN][32];
  curandStateXORWOW_t state _32;
  unsigned int rnd _32;
  int rnd_left _32;
  __WARP_INIT
  curand_init(seed, TID, 0, &state _);
  rnd_left _ = 0;
  ZERO(counts[blockIdx.x], unsigned long long, 1)
  __WARP_END
  for(int _loop = 0; _loop < loops; ++_loop) {
    __WARP_INIT
    ZERO(positions, unsigned int, 1)
    int checkpoint = 0;
    int current_checkpoint_idx = 0;
    for (int k = 0; k < T; ++k) {
      if (rnd_left _ == 0) {
        rnd _ = curand(&state _);
        rnd_left _ = 32;
      }
      checkpoint += (rnd _ & 1) == 0 ? -1 : 1;
      rnd _ >>= 1;
      --rnd_left _;
      if (k + 1 == CHECKPOINTS[current_checkpoint_idx]) {
        positions[current_checkpoint_idx][threadIdx.x] = (checkpoint / 2);
        ++current_checkpoint_idx;
      }
    }
    int target_idx = -1;
    for (int i = 0; i < TARGETS_LEN; ++i) {
      if (checkpoint / 2 == TARGETS[i]) {
        target_idx = i;
        break;
      }
    }
    if (target_idx >= 0) {
      for (int i = 0; i < CHECKPOINTS_LEN; ++i) {
        int x = positions[i][TID_LOCAL];
        if (x >= BINS_MIN && x <= BINS_MAX) {
            ++counts[blockIdx.x][TID_LOCAL][target_idx][i][x - BINS_MIN];
        }
      }
    }
    __WARP_END
  }
  __WARP_INIT
  for  (int i = 1; i < 32; ++i) {
    for (int j = 0; j <= sizeof(**counts) / sizeof(*****counts) / 32; j++) {
      int idx = j * 32 + TID_LOCAL;
      if (j * 32 + TID_LOCAL < sizeof(**counts) / sizeof(*****counts)){
        ((long long *) counts[blockIdx.x][0])[j * 32 + TID_LOCAL] += ((long long *) counts[blockIdx.x][i])[j * 32 + TID_LOCAL];
      }
    }
  }
  __SYNCTHREADS
  if (blockIdx.x == 0) {
    for (int b = 0; b < BLOCKS; ++b) {
      for (int i = 0; i <= sizeof(**counts) / sizeof(*****counts) / 32; ++i) {
        if (i * 32 + TID_LOCAL < sizeof(**counts) / sizeof(*****counts)) {
          out[i * 32 + TID_LOCAL] += ((long long *) counts[b][0])[i * 32 + TID_LOCAL];
        }
      }
    }
  }
  __WARP_END
}

int main() {
  unsigned long long out[TARGETS_LEN][CHECKPOINTS_LEN][BINS_LEN];
  run_simulation(5, 1111111111, (long long *) &out);
  1+1;
}