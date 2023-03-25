#include <cstdio>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    int cnt{0};
    cudaGetDeviceCount(&cnt);
    printf("Number of GPUs: %d\n", cnt);

    int version;
    cudaRuntimeGetVersion(&version);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("-------------------------------------\n");
    printf("Device name: %s\n", prop.name);
    printf("CUDA Runtime Version: %d.%d\n", version/1000, (version%100)/10);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total global memory: %ld bytes (%lf GiB)\n", prop.totalGlobalMem, prop.totalGlobalMem/1.074e+9);
    printf("-------------------------------------\n");

    cuda_hello<<<1, 10>>>(); 
    cudaDeviceSynchronize();
    return 0;
}
