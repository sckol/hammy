#ifdef USE_CUDA
/* CUDA path: real GPU warps, curand, no CPU emulation */
#include <curand_kernel.h>
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
