#ifndef FROM_PYTHON
#include "lattice.h"
#endif

STATIC_ASSERT(X_BINS_LEN == X_BINS_MAX - X_BINS_MIN + 1, x_bins_len_does_not_match);
STATIC_ASSERT(Y_BINS_LEN == Y_BINS_MAX - Y_BINS_MIN + 1, y_bins_len_does_not_match);
STATIC_ASSERT(BINS_FLAT_LEN == X_BINS_LEN * Y_BINS_LEN, bins_flat_len_does_not_match);
__device__ unsigned long long counts [BLOCKS][32][TARGETS_LEN][CHECKPOINTS_LEN][BINS_FLAT_LEN];
__EXTERN __global__ EXPORT void run_simulation(unsigned long long loops, const unsigned long long seed,  unsigned long long* out) {
  __shared__ int positions_x[CHECKPOINTS_LEN][32];
  __shared__ int positions_y[CHECKPOINTS_LEN][32];
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
    ZERO(positions_x, int, 1)
    ZERO(positions_y, int, 1)
    int pos_x = 0;
    int pos_y = 0;
    int current_checkpoint_idx = 0;
    for (int k = 0; k < T; ++k) {
      if (rnd_left _ < 2) {
        rnd _ = curand(&state _);
        rnd_left _ = 32;
      }
      int dim_bit = rnd _ & 1;
      rnd _ >>= 1;
      int step = (rnd _ & 1) == 0 ? -1 : 1;
      rnd _ >>= 1;
      rnd_left _ -= 2;
      if (dim_bit == 0) pos_x += step;
      else pos_y += step;
      if (k + 1 == CHECKPOINTS[current_checkpoint_idx]) {
        positions_x[current_checkpoint_idx][threadIdx.x] = pos_x / 2;
        positions_y[current_checkpoint_idx][threadIdx.x] = pos_y / 2;
        ++current_checkpoint_idx;
      }
    }
    int final_x = pos_x / 2;
    int final_y = pos_y / 2;
    int target_idx = -1;
    for (int i = 0; i < TARGETS_LEN; ++i) {
      if (final_x == TARGETS_X[i] && final_y == TARGETS_Y[i]) {
        target_idx = i;
        break;
      }
    }
    if (target_idx >= 0) {
      for (int i = 0; i < CHECKPOINTS_LEN; ++i) {
        int bx = positions_x[i][TID_LOCAL];
        int by = positions_y[i][TID_LOCAL];
        if (bx >= X_BINS_MIN && bx <= X_BINS_MAX && by >= Y_BINS_MIN && by <= Y_BINS_MAX) {
          int idx = (bx - X_BINS_MIN) * Y_BINS_LEN + (by - Y_BINS_MIN);
          ++counts[blockIdx.x][TID_LOCAL][target_idx][i][idx];
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
        ((unsigned long long *) counts[blockIdx.x][0])[j * 32 + TID_LOCAL] += ((unsigned long long *) counts[blockIdx.x][i])[j * 32 + TID_LOCAL];
      }
    }
  }
  __SYNCTHREADS
  if (blockIdx.x == 0) {
    for (int b = 0; b < BLOCKS; ++b) {
      for (int i = 0; i <= sizeof(**counts) / sizeof(*****counts) / 32; ++i) {
        if (i * 32 + TID_LOCAL < sizeof(**counts) / sizeof(*****counts)) {
          out[i * 32 + TID_LOCAL] += ((unsigned long long *) counts[b][0])[i * 32 + TID_LOCAL];
        }
      }
    }
  }
  __WARP_END
}

#ifndef FROM_PYTHON
#include <string.h>
int main() {
  unsigned long long out[TARGETS_LEN][CHECKPOINTS_LEN][BINS_FLAT_LEN];
  memset(out, 0, sizeof(out));
  run_simulation(5000, 1111111111, (unsigned long long *) &out);
}
#endif
