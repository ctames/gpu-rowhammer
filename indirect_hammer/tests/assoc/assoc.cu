#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "gputimer.h"

const size_t CACHESIZE = 1.5 * (1 <<20);

__global__ void multi_section_assoc_test(int* data, int way, int spread, int num_sections, int iterations) {
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	//uint nthreads = blockDim.x * gridDim.x;
	uint vals_per_section = CACHESIZE/4;
	
	int sum;
	for (int t = 0; t < iterations; t++) {
		for (int i = tid * vals_per_section; i < (tid * vals_per_section) + (way * spread); i += spread) {
			int n = data[i];
			sum += n;
		}
	}
	data[0] = sum;

}

int main(int argc, char** argv) {
    // commandd line params
    int way = atoi(argv[1]);
    int spread = atoi(argv[2]);
    int num_sections = atoi(argv[3]);
	int iterations = atoi(argv[4]); 
	 
    // host data
    int* dataH = (int*) malloc(num_sections*CACHESIZE);
    memset(dataH, 0, num_sections*CACHESIZE);
    srand(time(NULL));
    for (int i = 0; i < (num_sections*CACHESIZE)/4; i++) {
        dataH[i] = rand();
    }   

	// copy to device
    int* dataD;
    cudaMalloc((void**)&dataD, num_sections*CACHESIZE);
    cudaMemcpy(dataD, dataH, num_sections*CACHESIZE, cudaMemcpyHostToDevice);

    // timers
    GpuTimer timer1;
	timer1.Start();
	multi_section_assoc_test<<<1, num_sections>>>(dataD, way, spread, num_sections, iterations);
	timer1.Stop();
	printf("way: %d | spread: %d | sections: %d | iterations: %d | time: %g\n", way, spread, num_sections, iterations, timer1.Elapsed());

}


