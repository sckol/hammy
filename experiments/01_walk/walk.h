
#define T 1000
int TARGETS[] = {0,1,2,5,10};
#define TARGETS_LEN 5
int CHECKPOINTS[] = {100,200,300,400,500,600,700,800,900,1000};
#define CHECKPOINTS_LEN 10
#define BINS_MIN -50
#define BINS_MAX 50
#define BINS_LEN 101

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
#ifdef USE_CUDA

#else
/*
 * PCG Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     http://www.pcg-random.org
 */

/*
 * This code is derived from the full C implementation, which is in turn
 * derived from the canonical C++ PCG implementation. The C++ version
 * has many additional features and is preferable if you can use C++ in
 * your project.
 */

#ifndef PCG_BASIC_H_INCLUDED
#define PCG_BASIC_H_INCLUDED 1

#include <inttypes.h>

#if __cplusplus
extern "C" {
#endif

struct pcg_state_setseq_64 {    // Internals are *Private*.
    uint64_t state;             // RNG state.  All values are possible.
    uint64_t inc;               // Controls which RNG sequence (stream) is
                                // selected. Must *always* be odd.
};
typedef struct pcg_state_setseq_64 pcg32_random_t;

// If you *must* statically initialize it, here's one.

#define PCG32_INITIALIZER   { 0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL }

// pcg32_srandom(initstate, initseq)
// pcg32_srandom_r(rng, initstate, initseq):
//     Seed the rng.  Specified in two parts, state initializer and a
//     sequence selection constant (a.k.a. stream id)

void pcg32_srandom(uint64_t initstate, uint64_t initseq);
void pcg32_srandom_r(pcg32_random_t* rng, uint64_t initstate,
                     uint64_t initseq);

// pcg32_random()
// pcg32_random_r(rng)
//     Generate a uniformly distributed 32-bit random number

uint32_t pcg32_random(void);
uint32_t pcg32_random_r(pcg32_random_t* rng);

// pcg32_boundedrand(bound):
// pcg32_boundedrand_r(rng, bound):
//     Generate a uniformly distributed number, r, where 0 <= r < bound

uint32_t pcg32_boundedrand(uint32_t bound);
uint32_t pcg32_boundedrand_r(pcg32_random_t* rng, uint32_t bound);

#if __cplusplus
}
#endif

#endif // PCG_BASIC_H_INCLUDED

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

#endif