#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

// CARD TARGETED : K40c

// 1.5 MB 
const size_t CACHESIZE = 1.5 * (1<<20);

// 32 B
const size_t CLSIZE = 32;

const size_t intsize = sizeof(int);

void check_error(cudaError_t cudaerr) {
    if (cudaerr != cudaSuccess) {
        printf("FAILED WITH ERROR: \"%s\".\n", cudaGetErrorString(cudaerr));
        exit(-1);
    }   
}

__global__ void fill_cache_twice_strideCLSIZE(int* vals) {
	int sum; 
	for (int t = 0; t < 10000000; t++) {
	for (int i = 0; i < (2*CACHESIZE)/intsize; i += CLSIZE/intsize) {
		int n1 = vals[i];
		int n2 = vals[i+1];
		sum += n2 - n1;
	}}
	vals[0] = sum;
	//printf("first kernel\n");
}

__global__ void toggle_address(int* val) {
	int n1 = *val; 
	*(val++) = n1;
	//printf("second kernel\n");
}

int main() {
	srand(time(NULL));	

	int* valsHost = (int*) malloc(2*CACHESIZE); 
	
	memset(valsHost, 0, 2*CACHESIZE); 
	
	for (int i = 0; i < (2*CACHESIZE)/intsize; i++) {
		valsHost[i] = (int)rand();
	} 
	
	int* valsDevice;
 	cudaMalloc((void**)&valsDevice, 2*CACHESIZE);
 	cudaMemcpy(valsDevice, valsHost, 2*CACHESIZE, cudaMemcpyHostToDevice); 
 		
 	int* val = &valsDevice[CLSIZE/intsize];
 
 	fill_cache_twice_strideCLSIZE<<<1,1>>>(valsDevice); 
 	//check_error(cudaDeviceSynchronize());
	toggle_address<<<1,1>>>(val);
 	//check_error(cudaDeviceSynchronize());
	
}
