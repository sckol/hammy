#include <iostream>

using std::cout;
using std::endl;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__
void vecAddKernel(float const *a, float const *b, float *c, const unsigned int n) {
    printf("111\n");
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vecAdd(float const *a_h, float const *b_h, float *c_h, const unsigned int n) {
    const unsigned int size = n * sizeof(float);
    float *a_d, *b_d, *c_d;
    gpuErrchk(cudaMalloc((void **) &a_d, size));
    gpuErrchk(cudaMalloc((void **) &b_d, size));
    gpuErrchk(cudaMalloc((void **) &c_d, size));
    gpuErrchk(cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(c_d, c_h, size, cudaMemcpyHostToDevice));    
    vecAddKernel<<<ceil(n / 256.), 256>>>(a_d, b_d, c_d, n);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(a_d));
    gpuErrchk(cudaFree(b_d));
    gpuErrchk(cudaFree(c_d));    
}

int main(int argc, char **argv) {
    float const a[] {3, 12.5, 2, 90};
    float const b[] {5, 12.2, 2.1, 0};
    float c[] {0, 0, 0, 0};
    unsigned int n {sizeof(a)/sizeof(a[0])};
    vecAdd(a, b, c, n);  
    for (auto i = 0; i < n; ++i) {
        cout << c[i] << endl;
    }
}
