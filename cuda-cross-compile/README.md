# CUDA Cross-Compiling
* Version: 10.2

## Prerequisite
* CUDA Toolkit 10.2

## Install CUDA cross compiler
* CUDA Toolkit에는 기본적으로 크로스 컴파일러가 제공되지 않음
* Jetpack SDK에 포함되어있는 cuda cross compiler를 수동으로 다운로드
* L4T 32.7 버전 기준
```sh
sudo apt-add-repository "https://repo.download.nvidia.com/jetson/x86_64/bionic r32.7 main"
```
* CUDA 버전에 맞는 크로스 컴파일러 설치
```sh
sudo apt install cuda-cross-aarch64-10-2
```

## Build
```sh
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../aarch64_cross.cmake ..
make
```

---

## CLI
* Maxwell
```sh
nvcc -arch=sm_50 -ccbin=aarch64-linux-gnu-g++-7 -I/usr/local/cuda/include -L/usr/local/cuda/targets/aarch64-linux-gnu main.cu -o main
```

## References
* https://medium.com/trueface-ai/how-to-cross-compile-opencv-and-mxnet-for-nvidia-jetson-aarch64-cuda-99d467958bce
* https://docs.nvidia.com/jetson/jetpack/install-jetpack/index.html#jetpack-debian-packages
* https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
