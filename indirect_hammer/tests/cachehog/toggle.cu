#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "gputimer.h"

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


__global__ void toggle(int* data, int* vals, int num_vals, int* sum) {
	sum = 0;
	for (int i = 0; i < num_vals; i++) {
		sum += vals[i];
	}
}

int main(int argc, char** argv) {
	if (argc != 5) {
		printf("USAGE: ./toggle <# blocks: int> <# threads: int> <size_mult: double (multipler of cache size)> <num_vals: int>\n");
	}
	int blocks = atoi(argv[1]);
	int threads = atoi(argv[2]);
	double size_mult = atof(argv[3]);
	int num_vals = atoi(argv[4]);
	srand(time(NULL));	

	int size = (int)(size_mult * CACHESIZE);

	int* dataH = (int*) malloc(size); 
	int* valsH = (int*) malloc(num_vals*intsize);	
	
	for (int i = 0; i < size/intsize; i++) {
		dataH[i] = (int)rand();
	} 
	
	int* dataD;
 	cudaMalloc((void**)&dataD, size);
 	cudaMemcpy(dataD, dataH, size, cudaMemcpyHostToDevice); 

	// pick addresses 
	int num_ints = size/intsize;
	for (int i = 0; i < num_vals; i++) {
		valsH[i] = rand() % num_ints;
	}	

	// copy to device
	int* valsD;
 	cudaMalloc((void**)&valsD, num_vals*intsize);
 	cudaMemcpy(valsD, valsH, num_vals*intsize, cudaMemcpyHostToDevice); 
 	
	// arb value
	int* sumH = (int*)malloc(intsize);
	int* sumD;
	cudaMalloc((void**)&sumD, intsize);

	GpuTimer timer1;
	timer1.Start();
 	toggle<<<blocks, threads>>>(dataD, valsD, num_vals, sumD);
	//check_error(cudaDeviceSynchronize());
	timer1.Stop();
	cudaMemcpy(sumH, sumD, intsize, cudaMemcpyDeviceToHost);
	printf("blocks: %d | threads: %d  | size_mult: %f | num_vals: %d/n", blocks, threads, size_mult, num_vals);
	printf("timer1: %g\n", timer1.Elapsed());
	printf("%d\n", sumH);
}
