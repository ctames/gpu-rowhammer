//#ifndef HAMMERTESTS_H
//#define HAMMERTESTS_H

#include <hammerutils.h>
#include <gputimer.h> 

int cachesize = 1.5 * (1 << 6);

// pointers[0] = host
// pointers[1] = device

// patterns: 1 = all 1's, 2 = all 0's, 3 = alternating, 4 = random

// CPU CODE

int load_xmb_single() {
	printf("Input #blocks, #threads, cachesize multiplier, stride, toggles, repetitions, data pattern\n");
	int blocks, threads, stride, pattern, iterations, reps;
	double mult; 
	cin >> blocks;
	cin >> threads;
	cin >> mult;
	cin >> stride;
	cin >> toggles;
	cin >> reps;
	cin >> pattern;
	
	size_t size = mult * cachesize;
	int** pointers = (int**) malloc(sizeof(int*)*2);
	pointers = mem_setup(size, pattern);

	for (int i = 0; i < reps; i++) {
		printf("launching test %d\n", i);
		GpuTimer timer;
		timer.Start();	
		lxs<<<blocks, threads>>>(pointers[1], size, stride, toggles); 
		timer.Stop();
		printf("test took %g\n", timer.Elapsed());
		mem_check(pointers[0], pointers[1], size);
	}
}

int load_xmb_multi() {

}

int assoc() {

}

int hog() {

}

// GPU KERNELS

__global__ void lxs(int* vals, size_t size, int stride, int toggles) {
    uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    int sum;
    for (int t = 0; t < toggles; t++) { 
        for (int i = tid; i < size/intsize; i += (i*stride)) {
            int n1 = vals[i];
			int n2 = vals[i+1];
			sum += n2 - n1;
		}
	}
	vals[0] = sum;          
}

//#endif
