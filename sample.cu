#include <stdio.h>
#include "gputimer.h"
__global__ void mykernel(int *data){
//for(int i = 0; i < 100000; i++)
for(int j = 0; j < 10; j++)
for(int k = 0; k < 100000; k++)
for(int l = 0; l < 100000; l++)
  data[l]++;
}

int main(){
  GpuTimer timer;
  int *d_data;
  cudaMalloc((void **)&d_data, sizeof(int)  * 268435456);
  //cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);
  timer.Start();
  mykernel<<<1,1>>>(d_data);
  timer.Stop();
  //cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
  //printf("data = %d\n", h_data);
  printf("Time taken on GPU: %g ms\n", timer.Elapsed());
  return 0;
}
