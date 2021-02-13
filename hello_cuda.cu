#include <iostream>

#include <stdio.h>

__global__ void myfirstkernel() {
}

int main() {
  myfirstkernel << <1,1 >> >();
  printf("Hello, CUDA!\n");
  return 0;
}
