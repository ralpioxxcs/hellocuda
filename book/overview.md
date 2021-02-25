## CUDA 프로그램 구조
* cuda 프로그램은 host와 gpu device에서 실행되는 프로그램
* device에서 호출되는 함수는 `__global__`의 키워드를 갖고있고 NVCC 컴파일러에 의해서 컴파일되어진다.

## Kernel call
ANSI-C 스타일의 cuda 디바이스 코드는 **kernel call**에 의해 host code로부터 시작된다.  
**kernel call**은 병렬처리를 위한 많은 수의 block, thread를 생성한다
```
kernel << < number of blocks, number of threads per block, size of shared memory > >> (parameters for kernerl)
```
`__global__` 키워드를 사용해 커널함수를 정의해야만 한다.kernel launch operator인 `<< < > >>`로 커널실행에 필요한 파라미터들을 구성한다.
* 실행하고자 하는 block의 갯수
* 각 block이 갖는 thread의 갯수 
* 공유메모리의 크기
이렇게 3가지의 parameters를 포함한다. 예를들면,
```cuda
gpuAdd << <1,1> >> (1, 4, d_c)
```
위의 코드는 1개의 block에서 1개의 thread로 수행된다.  
