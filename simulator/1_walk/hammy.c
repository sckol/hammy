#include "pcg_basic.h"

#define T 1000
#define LOOPS 23200
int TARGETS[] = {0,1,2,5,10};
#define TARGETS_LEN 5
int CHECKPOINTS[] = {100,200,300,400,500,600,700,800,900,1000};
#define CHECKPOINTS_LEN 10
#define BINS_MIN -50
#define BINS_MAX 50
#define BINS_LEN 101
#define STATIC_ASSERT(COND,MSG) typedef char static_assertion_##MSG[(COND)?1:-1]
STATIC_ASSERT(BINS_LEN == BINS_MAX - BINS_MIN + 1, bins_len_does_not_match);
 
#define __EXTERN
#define __global__
#define __shared__
#define _32 [32]
#define _ [threadIdx.x]
typedef pcg32_random_t curandStateXORWOW_t;
void curand_init(unsigned long long seed, unsigned long long sequence, unsigned long long offset, curandStateXORWOW_t *state) {
  pcg32_srandom_r(state, seed + sequence, offset);
}
unsigned int curand(curandStateXORWOW_t *state) {
  return pcg32_random_r(state);
}
struct ThreadCounter {
  int x;
  int y;
  int z;
};
void atomicAdd(unsigned long long int* address, unsigned long long int val) {
    *address += val;
}
#define __WARP_INIT for(struct ThreadCounter threadIdx = {0, 0, 0}; threadIdx.x < 32; ++threadIdx.x) { struct ThreadCounter blockIdx = {0, 0, 0}; struct ThreadCounter blockDim = {0, 0, 0};
#define __SYNCTHREADS } __WARP_INIT
#define __WARP_END }

#define TID (threadIdx.x + blockIdx.x * blockDim.x)
#define TID_LOCAL threadIdx.x
#define ZERO(arr, type, aligned) for (int _i = 0; _i + aligned <= sizeof(arr) / sizeof(type) / 32; ++_i) {   if (aligned || (_i * 32 + TID_LOCAL < sizeof(arr) / sizeof(type))) {     ((type *) arr)[_i * 32 + TID_LOCAL] = 0;   } }
__EXTERN __global__ void run(const unsigned long long seed,  unsigned long long* out) {
  __shared__ int positions[CHECKPOINTS_LEN][32];
  __shared__ unsigned long long counts [32][TARGETS_LEN][CHECKPOINTS_LEN][BINS_LEN];
  curandStateXORWOW_t state _32;
  unsigned int rnd _32;
  int rnd_left _32;
  __WARP_INIT
  curand_init(seed, TID, 0, &state _);
  rnd_left _ = 0;
  ZERO(counts, unsigned long long, 1)
  __WARP_END
  for(int _loop = 0; _loop < LOOPS; ++_loop) {
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
        positions[current_checkpoint_idx] _ = (checkpoint / 2);
        ++current_checkpoint_idx;
      }
    }    
    int target_idx = -1;
    for (int i = 0; i < TARGETS_LEN; ++i) {
      if (checkpoint == TARGETS[i]) {
        target_idx = i;
        break;
      }
    }
    if (target_idx >= 0) {
      for (int i = 0; i < CHECKPOINTS_LEN; ++i) {
        int x = positions[i][TID_LOCAL];
        if (x >= BINS_MIN && x <= BINS_MAX) {
            ++counts[TID_LOCAL][target_idx][i][x - BINS_MIN];
        }
      }
    }
    __WARP_END
  }
  __WARP_INIT
  for  (int i = 1; i < 32; ++i) {
    for (int j = 0; j <= sizeof(*counts) / sizeof(****counts) / 32; j++) {
      int idx = j * 32 + TID_LOCAL;
      if (j * 32 + TID_LOCAL < sizeof(*counts) / sizeof(***counts)) {
        ((int *) counts[0])[j * 32 + TID_LOCAL] += ((int *) counts[i])[j * 32 + TID_LOCAL];
      }
    }
  }
  __SYNCTHREADS
  for (int i = 0; i <= sizeof(*counts) / sizeof(****counts) / 32; ++i) {
    if (i * 32 + TID_LOCAL < sizeof(*counts) / sizeof(****counts)) {
      out[i * 32 + TID_LOCAL] = ((long long *) counts[0])[i * 32 + TID_LOCAL];
    }
  }  
  __WARP_END
}




int main() {
  unsigned long long out[TARGETS_LEN][CHECKPOINTS_LEN][BINS_LEN];
  run(1111111111, (long long *) &out);
  1+1;
}