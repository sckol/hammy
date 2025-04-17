#define __EXTERN
#define __global__
#define __shared__
#define __device__
#define _32 [32]
#define _ [threadIdx.x]
typedef pcg32_random_t curandStateXORWOW_t;
void curand_init(unsigned long long seed, unsigned long long sequence, unsigned long long offset, curandStateXORWOW_t *state) {
  pcg32_srandom_r(state, offset, (((unsigned long long)(seed & 0xFFFFFFFF) << 32) | (unsigned long long)(sequence & 0xFFFFFFFF)));
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
