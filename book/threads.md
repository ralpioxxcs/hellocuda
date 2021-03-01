# CUDA Thread

## Threads
CUDA는 병렬처리 관점에서 계층적인 구조를 갖는다. **kernel execution**은 여러 block에서 수행된다.  
* 1개의 thread당 1개의 block의 구성으로 여러개의 block
* 여러개의 thread를 갖는 1개의 block

같은 block에 있는 thread들은 공유메모리를 통해 서로 데이터를 공유할 수 있다. 따라서 1개의 block에서 생성된 여러개의 thread들은 서로 값을 공유할 수 있다. 

block 1개당 실행할 수 있는 thread의 갯수를 나타내는 속성인 `maxThreadPerBlock`을 통해 확인할 수 있으며 보통 **512**, **1024**의 값을 갖는다. 또한 한번에 실행가능한 최대 block의 갯수는 **65535**이다.

한 block당 여러 thread를 실행하는 방법 혹은 한 thread를 갖는 여러 block 대신에 여러 block에서 여러개의 thread를 실행하는 방법이 이상적이다. 

에를들어 `N=50000`의 thread를 실행시켜, vector add 병렬연산을 하는 함수가 있다고 해보자.
```
gpuAdd<< <( (N+511)/512), 512 > >> (d_a, d_b, d_c)
```
한 block당 실행가능한 최대 thread의 갯수는 **512** 이고, 총 block갯수는 실행시킬 총 thread 갯수를 512로 나눈것과 같다. 만약 N값이 512로 딱 나누어 떨어지지 않는다면, 총 block갯수는 실제값보다 낮게 잘못된 값이 될 수있다. 따라서 511을 더해준뒤, 512로 나눠주어야 한다.

하지만, 이런 방식이 모든 N에 대하여 적용되지 않으므로 적절한 block, thread의 갯수를 가질수 있도록 커널 코드의 수정이 필요하다.
```cpp
#include "stdio.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 50000
__global__ void gpuAdd(int *d_a, int *d_b, int *d_c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while(tid < N) {
        d_c[tid] = d_a[tid] + d_b[tid];
        tid += blockDim.x * gridDim.x;
    }
}
```


|  |   |  |  |   |
|---------|----------|----------|----------|---|
| BLOCK 1 | Thread 0 | Thread 1 | Thread 2 |   |
| BLOCK 2 | Thread 0 | Thread 1 | Thread 2 |   |
| BLOCK 3 | Thread 0 | Thread 1 | Thread 2 |   |

3개의 Block과 각 block당 3개의 Thread를 갖는 연산이 있다고 하자  
각 tid(task ID)값은 `tid = threadId.x + blockIdx.x * blockDim.x`을 통해서 계산되어진다. 여기서 `blockDim.x`은 block당 thread의 갯수를 나타낸다. 위와 같이 계산하면 고유한 tid값들을 가질 수 있게된다.  
각 dimension당 block갯수를 뜻하는 `gridDim.x`값을 통해 offset을 설정하고 `blockDim.x * gridDim.x` 으로 

여기서는 1 dimension block을 정의했으므로, `gridDim.x`의 값은 512을 갖는다.

---

## Memeory architecture
GPU는 각각 다른 속도, 용도를 갖는 메모리공간으로 이루어져있다. 이 메모리공간은 전역 메모리, 공유 메모리, 지역 메모리 등등 계층적으로 서로 다른 메모리 chunk들로 나뉘어져있으며 프로그램이 동작하는 방법에 따라 접근되는 방법이 서로 다르다.

## Anatomy
각 thread는 자기 소유의 local memory, register를 갖고있다. CPU와 다르게 GPU는 데이터를 저장하기 위해 수많은 register들을 갖고있다. register file은 가장 빠른 메모리이며, 같은 block에 있는 thread들은 block내의 모든 thread에서 접근 가능한 공유 메모리를 갖고있다. 모든 block, thread에서 접근 가능한 전역 메모리(global memory)는 가장 큰 메모리이자, 높은 latency를 갖는다. caching을 위한 L1, L2 캐시도 있고, 이외에도 constant, texture memory등이 있다.

### Global memory
global memory는 gpu내의 모든 block, thread에서 접근 및 쓰기가 가능하지만, 느리다는 단점을 갖는다. `cudaMalloc`을 통해 할당된 메모리들은 모두 global memory로 할당되어진다.
```cpp
#include <stdio.h>
#define N 5

__global__ void gpu_global_memory(int *d_a) {
    d_a[threadIdx.x] = threadIdx.x;
}

int main(int argc, int **argv) {
    int h_a[N];
    int *d_a;
    cudaMalloc((void **)&d_a, sizeof(int) *N);)
    cudaMemcpy((void *)d_a, (void *)h_a, sizeof(int) *N, cudaMemcpyHostToDevice);
    gpu_global_memory << <1,N> >>(d_a);
    cudaMemcpy((void *)h_a, (void *)d_a, sizeof(int) *N, cudaMemcpyDeviceToHost);
    printf("Array in global memory : \n");
    for(int i=0; i < N; i++) {
        printf("At index: %d --> %d \n", i, h_a[i]);
    }
    return 0;
}
```

위 코드처럼 Host에서 `cudaMalloc`을 통해 (d_a)메모리가 할당되어지고, 이 메모리를 가르키는 포인터가 kernel function의 parameter로 넘겨진다.  

### Local memory & registers
kernel 변수들을 register에 담기에 충분한 공간이 없을때 local memory를 사용하게 되는데 이를 **register spilling**이라고 부른다. local memory는 global memory의 일부분이며, register에 비해 느린 접근속도를 갖는다. local memory가 L1, L2에 cache되어지긴 하지만, register spilling은 프로그램에 안좋은 영향을 끼치지 않을지도 모른다(?)
```cpp
#include <stdio.h>
#define N 5

__global__ void gpu_local_memory(int d_in) {
    int t_local;
    t_local = d_in * threadIdx.x;
    printf("Value of local variable in current thread is : %d \n", t_local);
}

int main(int argc, int **argv) {
    printf("Use of local memory on GPU:\n");
    gpu_local_memory << <1,N> >>(5);
    cudaDeviceSynchronize();
    return 0;
}
```
여기서 `t_local`변수는 각 thread에서 register에 저장되어진다. kernel function에서 이 변수가 사용되어질때, computation속도는 가장 빠를것이다.

### Cache memory
최신 GPU에는 멀티프로세서 하나당 L1캐시, L2캐시가 있다. L1캐시가 thread 실행에 있어 가장 가까우므로 가장 빠르다. L1캐시와 shared memory는 64KB의 같은 크기를 사용한다.ㄴ

---

## Thread synchronization
실제로 thread가 실행되고 그 연산 결과를 다른 thread로 전달하지 않는경우는 거의 없다. 이런 각 thread끼리의 연산결과를 공유하기위해 **shared memory**라는것이 있다. 많은 thread가 병렬로 수행되고 동일한 memory 위치에서 읽기와 쓰기 연산을 할떄 모든 thread끼리는 동기화작업이 필수적이어야한다.

### Shraed memory  

### Atomic operations  

### Constant memory  

### Texture memory  


---
## Practical examples
### Dot product



