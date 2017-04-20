#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "gputimer.h"

const size_t CACHESIZE = 1.5 * (1 << 20);

__global__ void readwrite_test(int* data, int stride, int size, int iterations) {
	/*
		actual implementation goes here
	*/

	int sum = 0;
	for (int t = 0; t < iterations; t++) {
		int n = data[4];
		sum += n;
		data[4] = sum/2; 
	}

}

int main(int argc, char** argv) {
    // commandd line params
    int blocks = atoi(argv[1]);
	int threads = atoi(argv[2]); 
	int stride = atoi(argv[3]);
    double size_mult = atof(argv[4]);
    int iterations = atoi(argv[5]);
    
	int size = size_mult * CACHESIZE;
    
	// host data
	int* dataH = (int*) malloc(size);
    memset(dataH, 0, size);
    srand(time(NULL));
    for (int i = 0; i < (size)/4; i++) {
        dataH[i] = rand();
    }
	
	 // copy to device
    int* dataD;
    cudaMalloc((void**)&dataD, size);
    cudaMemcpy(dataD, dataH, size, cudaMemcpyHostToDevice);

    // timers
    GpuTimer timer1;
    timer1.Start();
	readwrite_test<<<blocks, threads>>>(dataD, stride, size, iterations);
	timer1.Stop();
	printf("blocks: %d | threads: %d | stride: %d | sizemult: %f | iterations: %d | time: %g\n", blocks, threads, stride, size_mult, iterations, timer1.Elapsed());
}
