#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

// cache size constant for convenience/readability 
const size_t CACHESIZE = 1.5 * (1<<20); 

// function to handle CUDA API call error val returns
void check_error(cudaError_t cudaerr) {
    if (cudaerr != cudaSuccess) {
        printf("FAILED WITH ERROR: \"%s\".\n", cudaGetErrorString(cudaerr));
        exit(-1);
    }   
}

// hammer attempt test kernel
__global__  void hammer_attempt(int* data, int* indices, int* eviction, int iterations, int numvals, int stride) {
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint nthreads = blockDim.x * gridDim.x;
	
	int sum;
	for (int j = 0; j < iterations; j++) {	
	
			
		if (tid < 8) {
			int index = indices[tid];
			int n = data[index];
			sum += (n << 1); 
		}
		
		
		
		/*
		if (tid == 0) {
			int n1 = data[1413];
			int n2 = data[56102];
			int n3 = data[101336];
			int n4 = data[169659];
			int n5 = data[199685];
			int n6 = data[272110];
			int n7 = data[309174];
			int n8 = data[368505];
			sum = sum + n1 - n2 + n3 - n4 + n5 - n6 + n7 - n8;
		}
		*/
		
		//__syncthreads();	
		for (int i = (tid * stride) + 393216; i < 393216*2.5; i += (nthreads * stride)) {
			int n = data[i];
			sum -= (n >> 1);
		//	__syncthreads();
		}

		/*
		for (int i = 393216; i < 393216*3; i+=8) {
			int n = data[i];
			sum -= (n >> 1); 
		}
		*/
		
		/*
		for (int i = tid; i < numvals; i += nthreads) {
			int n = data[eviction[i]];
			sum -= (n >> 1); 
		}
		*/
		
	}
}

int main(int argc, char** argv) {
	// command line params, seed rand
	int blocks = atoi(argv[1]);
    int threads = atoi(argv[2]);
    int stride = atoi(argv[3]);
	int iterations = atoi(argv[4]);
	srand(time(NULL));

	// host array of random vals
	int* data_host = (int*) malloc(2.5*CACHESIZE);
	for (int i = 0; i < (2.5*CACHESIZE)/4; i++) {
		data_host[i] = rand(); 
	}

	// copy to device
	int* data_device;
	check_error(cudaMalloc((void**)&data_device, 2.5*CACHESIZE));
	check_error(cudaMemcpy(data_device, data_host, 2.5*CACHESIZE, cudaMemcpyHostToDevice));

	// host array for addresses
	// pick addresses in first 1.5MB of data
	// - explanation of values -
	// 393216: number of integers containted in 1.5MB cache
	// 49512: number of integers in a 1/8 slice of cache
	// 24576: number of integers in a 1/16 slice of cache
	// address choosing process: pick address in first 1/16, skip next 1/16, pick address in next 1/16...
	int* indices_host = (int*)malloc(sizeof(int)*8); 	 
	for (int i = 0; i < 8; i++) {
		int sector_start = i * 49512; 
		indices_host[i] = (rand() % 24576) + sector_start;
		printf("%d\n", indices_host[i]);
	}

	// copy to device
	int* indices_device;
	check_error(cudaMalloc((void**)&indices_device, sizeof(int)*8));
	check_error(cudaMemcpy(indices_device, indices_host, sizeof(int)*8, cudaMemcpyHostToDevice));

	// host array of indices for eviction purposes
	int numvals = (393216*2) / stride;
	int* eviction_host = (int*)malloc(sizeof(int)*numvals);
	for (int i = 0; i < numvals; i++) {
		eviction_host[i] = 393216 + (i * stride);
	}

	// copy to device
	int* eviction_device;
	check_error(cudaMalloc((void**)&eviction_device, sizeof(int)*numvals));
	check_error(cudaMemcpy(eviction_device, eviction_host, sizeof(int)*numvals, cudaMemcpyHostToDevice));

	// launch kernel 
	hammer_attempt<<<blocks, threads>>>(data_device, indices_device, eviction_device, iterations, numvals, stride);
	check_error(cudaDeviceSynchronize()); 
}
