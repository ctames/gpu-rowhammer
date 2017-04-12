#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "gputimer.h"

const size_t CACHESIZE = 1.5 * (1<<20);

// TEST 1, 2
__global__ void evict(int* data, int spread, int toggle_val) {
	int val;
	uint tid = threadIdx.x;
	int thread_val = toggle_val + (CACHESIZE/4) + (tid*spread);
	val = data[thread_val];
	data[thread_val+1] = val / 2;	
}

// TEST 1, 2
__global__ void toggle(int* data, int toggle_val) {
	int val = data[toggle_val];
	data[toggle_val+1] = val / 2; 
}

// TEST 3
__global__ void evict_toggle_repeat(int* data, int spread, int toggle_val, int toggles) {
	int val, sum; 
	uint tid = threadIdx.x;
	int thread_val = toggle_val + (CACHESIZE/4) + (tid*spread);
	for (int i = 0; i < toggles; i++) {
		val = data[thread_val];
		sum += val;
		// memory fence
		__threadfence_system();
		if (tid == 0) {
			val = data[toggle_val];
			data[toggle_val+1] = val / 2;
		}
	}
}

// TEST 4
__global__ void toggle_evict_toggle(int* data, int spread, int toggle_val) {
	int val = 0;
	uint tid = threadIdx.x;
	int thread_val = toggle_val + (CACHESIZE/4) + (tid*spread);

	// toggle
	if (tid == 0) {
		int val1 = data[toggle_val];
		val = val1*2;
	}
	
	// evict
	int val2 = data[thread_val];
	val = val/val2; 

	// toggle
	if (tid == 0) {
		int val1 = data[toggle_val];
		val += val1;
		//data[toggle_val+1] = val;
	}
}

// TEST 5
__global__ void global_accesses(int* data, int spread, int toggle_val, int* val) {
	uint tid = threadIdx.x;
	int n1 = data[0];
	*val = n1*2; 
}

// TEST 6
// change multiplier of cache size to large for this test
// 393216 = num ints in cachesize
__global__ void multi_cachesize_spaced_accesses(int* data, int* val) {
	int n;
	for (int t = 0; t < 1; t++) {
	for (int i = 0; i < 21; i++) {
		int start = i * 393216;
		for (int j = 0; j < 768; j++) {
			n = data[start + (j * 16)];
			val += (n >> 2); 
		}
	}
	}
} 

int main(int argc, char** argv) {
	// commandd line params
	int way = atoi(argv[1]);
	int spread = atoi(argv[2]);
	int toggle_val = atoi(argv[3]);
	int toggles = atoi(argv[4]);
	
	// host data
	int* dataH = (int*) malloc(40*CACHESIZE);
	memset(dataH, 0, 40*CACHESIZE);
	srand(time(NULL));
	for (int i = 0; i < (40*CACHESIZE)/4; i++) {
		dataH[i] = rand();
	}

	// copy to device
	int* dataD;
	cudaMalloc((void**)&dataD, 40*CACHESIZE);
	cudaMemcpy(dataD, dataH, 40*CACHESIZE, cudaMemcpyHostToDevice);

	// timers
	GpuTimer timer1;
	GpuTimer timer2;

	// allocate space for arbitrary value to print to prevent optimizations	
	int* valH = (int*) malloc(sizeof(int));
	int* valD;
	cudaMalloc((void**)&valD, sizeof(int));
	cudaMemcpy(valD, valH, sizeof(int), cudaMemcpyHostToDevice);

	// TEST 1 : time single evict and toggle call seperately
	/*
	timer1.Start();
	evict<<<1, way>>>(dataD, spread, toggle_val);
	timer1.Stop();
	timer2.Start();
	toggle<<<1,1>>>(dataD, toggle_val);
	timer2.Stop();
	printf("timer1: %g | timer2: %g\n", timer1.Elapsed(), timer2.Elapsed());
	*/

	// TEST 2 : time and profile iterative calls
	/*
	timer1.Start();	
	for (int i = 0; i < 10; i++) {
		evict<<<1, way>>>(dataD, spread, toggle_val);
		toggle<<<1,1>>>(dataD, toggle_val);
	}
	timer1.Stop();
	printf("timer1: %g | timer2: %g\n", timer1.Elapsed(), timer2.Elapsed());
	*/

	// TEST 3: time and profile evict_toggle as one single kernel launch (iterative accesses done in kernel)
	/*
	timer1.Start();
	evict_toggle_repeat<<<1, way>>>(dataD, spread, toggle_val, toggles);
	timer1.Stop();
	printf("timer1: %g | timer2: %g\n", timer1.Elapsed(), timer2.Elapsed());
	*/

	// TEST 4: toggle once, attempt eviction, toggle again in single kernel
	// for ease of profiling approach's l2 hits, throughput, and read transactions
	/*
	timer1.Start();	
	toggle_evict_toggle<<<1, way>>>(dataD, spread, toggle_val);
	timer1.Stop();
	printf("timer1: %g | timer2: %g\n", timer1.Elapsed(), timer2.Elapsed());
	*/

	// TEST 5: trying to figure out how simple global acccesses contribute to metrics
	/*
	global_accesses<<<1, way>>>(dataD, spread, toggle_val, valD);
	cudaMemcpy(valH, valD, sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d\n", *valH);
	*/

	// TEST 6
	multi_cachesize_spaced_accesses<<<1, way>>>(dataD, valD);
	cudaMemcpy(valH, valD, sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d\n", *valH);
}
