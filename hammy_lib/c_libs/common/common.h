#ifdef USE_CUDA
#include <curand.h>
#else
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