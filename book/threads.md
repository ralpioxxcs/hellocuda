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
```
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

## Mmeory architecture
