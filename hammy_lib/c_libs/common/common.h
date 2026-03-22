#ifdef USE_CUDA
/* CUDA path: real GPU warps, PCG32 RNG (same as CPU — no curand dependency),
   no CPU emulation. Avoids #include <curand_kernel.h> which requires nvcc. */

/* PCG32 RNG — inline implementation for GPU (matches pcg_basic on CPU) */
struct __align__(8) pcg_state_setseq_64 {
    unsigned long long state;
    unsigned long long inc;
};
typedef struct pcg_state_setseq_64 pcg32_random_t;
typedef pcg32_random_t curandStateXORWOW_t;

__device__ __forceinline__ void pcg32_srandom_r(pcg32_random_t* rng, unsigned long long initstate, unsigned long long initseq) {
    rng->state = 0ULL;
    rng->inc = (initseq << 1u) | 1u;
    /* advance once */
    rng->state = rng->state * 6364136223846793005ULL + rng->inc;
    rng->state += initstate;
    rng->state = rng->state * 6364136223846793005ULL + rng->inc;
}

__device__ __forceinline__ unsigned int pcg32_random_r(pcg32_random_t* rng) {
    unsigned long long oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    unsigned int xorshifted = (unsigned int)(((oldstate >> 18u) ^ oldstate) >> 27u);
    unsigned int rot = (unsigned int)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

__device__ __forceinline__ void curand_init(unsigned long long seed, unsigned long long sequence, unsigned long long offset, curandStateXORWOW_t *state) {
    pcg32_srandom_r(state, offset, (((unsigned long long)(seed & 0xFFFFFFFF) << 32) | (unsigned long long)(sequence & 0xFFFFFFFF)));
}

__device__ __forceinline__ unsigned int curand(curandStateXORWOW_t *state) {
    return pcg32_random_r(state);
}

/* EXPORT: marks function as visible in shared library for CFFI/dlopen.
   Not needed on CUDA (kernels are called via CUDA runtime). */
#define EXPORT
/* CUDA-native macros: real per-thread storage, real warps, real barriers */
#define __EXTERN extern "C"
#define _32
#define _
#define __WARP_INIT
#define __SYNCTHREADS __syncthreads();
#define __WARP_END
#else
/* CPU path: emulate one 32-thread warp as a sequential for-loop */
#define BLOCKS 1
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
#endif
#define TID (threadIdx.x + blockIdx.x * blockDim.x)
#define TID_LOCAL threadIdx.x
#define ZERO(arr, type, aligned) for (int _i = 0; _i + aligned <= sizeof(arr) / sizeof(type) / 32; ++_i) { \
  if (aligned || (_i * 32 + TID_LOCAL < sizeof(arr) / sizeof(type))) { \
    ((type *) arr)[_i * 32 + TID_LOCAL] = 0; \
  } \
}
#define STATIC_ASSERT(COND,MSG) typedef char static_assertion_##MSG[(COND)?1:-1]
