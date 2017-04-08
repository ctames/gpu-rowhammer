//#ifndef HAMMERUTILS_H
//#define HAMMERUTILS_H

#include <unistd.h>
#include <stdlib.h> 
#include <time.h>

void check_error(cudaError_t cudaerr) {
    if (cudaerr != cudaSuccess) {
        printf("FAILED WITH ERROR: \"%s\".\n", cudaGetErrorString(cudaerr));
        exit(-1);
    }   
}

/*
 * allocates host mem
 * initializes according to desired pattern
 * allocates device mem
 * memcpys over
 * returns 2 element int* array, [0] points to host, [1] to device
 */

int** mem_setup(size_t size, int pattern) {
	int** pointers = (int**) malloc(sizeof(int*)*2); 
	
	int* host = (int*) malloc(size); 
	
	// 1 = all 1's
	// 2 = all 0's
	// 3 = alternate bytes of 1's and 0's
	// 4 = random
	switch(pattern) {
		case 1: {
			for (int i = 0; i < size/4; i++) {
				memset(host[i], 0xff, 4);	
			}
		}
		case 2: {
			for (int i = 0; i < size/4; i++) {
				memset(host[i], 0x00, 4);	
			}
		}
		case 3: {
			for (int i = 0; i < size/4; i++) {
				memset(host[i], 0xf0, 4);	
			}
		}
		case 4: {
			srand(time(NULL));
			for (int i = 0; i < size/4; i++) {
				host[i] = rand(); 
			}
		}
	}

	int* device;
	check_error(cudaMalloc((void**)&device, size));
	check_error(cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));

	pointers[0] = host;
	pointers[1] = device;

	return pointers;
}

/*
 * allocates a second host mem
 * copies device mem to that
 * compares host mem 1 and 2
 * reports discrepancies (bit flips)
 */

int** mem_check(int* host_pre, int* device, int size) {
	int flips = 0;
	int* host_post = (int*) malloc(size);
	check_error(cudaMemcpy(host_post, device, size, cudaMemcpyDeviceToHost)); 
	for (int i = 0; i < size/4; i++) {
		if (host_pre[i] != host_post[i]) {
			flips++;
			printf("bit flip found at index %d | pre val %d | post val %d\n", i, host_pre[i], host_post[i]); 
		} 
	}
	printf("number of bit flips: %d\n", flips);
}

//#endif 
