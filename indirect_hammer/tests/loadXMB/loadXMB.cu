#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "gputimer.h"

// CARD TARGETED : K40c

// 1.5 MB 
const size_t CACHESIZE = 1.5 * (1<<20);

// 32 B
// const size_t CLSIZE = 32;

const size_t intsize = sizeof(int);

void check_error(cudaError_t cudaerr) {
    if (cudaerr != cudaSuccess) {
        printf("FAILED WITH ERROR: \"%s\".\n", cudaGetErrorString(cudaerr));
        exit(-1);
    }   
}

__global__ void fill_cache_stride(int* vals, int size, int stride, int iters) {
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint nthreads = blockDim.x * gridDim.x; 
	int sum;
	for (int t = 0; t < iters; t++) { 
		for (int i = tid*stride; i < size/intsize; i += (nthreads*stride)) {
			int n1 = vals[i];
			//int n2 = vals[thread_i+1];
			sum += n1;
		}
		vals[0] = sum;
		//printf("first kernel\n");
	}
}

__global__ void fill_cache_stride_1thread(int* vals, int size, int stride) {
	//uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	//uint nthreads = blockDim.x * gridDim.x;

	int sum;
	for (int j = 0; j < 100; j++) {
	for (int i = 0; i < size/intsize; i += stride) {
		int n1 = vals[i];
		//int n2 = vals[thread_i+1];
		sum += n1;
	}}
	vals[0] = sum;
	//printf("first kernel\n");

}

int main(int argc, char** argv) {
	if (argc != 6) {
		printf("USAGE: ./loadXMB <# blocks: int> <# threads: int> <size_mult: double (multipler of cache size)> <stride: int> <iterations: int>\n");
	}
	int blocks = atoi(argv[1]);
	int threads = atoi(argv[2]);
	double size_mult = atof(argv[3]);
	int stride = atoi(argv[4]);
	int iters = atoi(argv[5]);
	srand(time(NULL));	

	int size = (int)(size_mult * CACHESIZE);

	int* valsHost = (int*) malloc(size); 
	
	memset(valsHost, 0, size); 
	
	for (int i = 0; i < size/intsize; i++) {
		valsHost[i] = (int)rand();
	} 
	
	int* valsDevice;
 	cudaMalloc((void**)&valsDevice, size);
 	cudaMemcpy(valsDevice, valsHost, size, cudaMemcpyHostToDevice); 
 		
	GpuTimer timer1;
	timer1.Start();
 	fill_cache_stride<<<blocks, threads>>>(valsDevice, size, stride, iters); 
 	//fill_cache_stride_1thread<<<1, 1>>>(valsDevice, size, stride); 
 	check_error(cudaDeviceSynchronize());
	timer1.Stop();
 	//check_error(cudaDeviceSynchronize());
	printf("blocks: %d | threads: %d  | size_mult: %f | stride: %d | iters: %d\n", blocks, threads, size_mult, stride, iters);
	printf("timer1: %g | val: %d\n", timer1.Elapsed(), valsHost[0]); 
}
