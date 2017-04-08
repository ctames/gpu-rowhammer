#include <stdio.h>
#include <iostream>
#include "gputimer.h"
#include <stdlib.h>
#include <unistd.h>

// CACHE SIZE
size_t cachesize = 1.5 * (1 << 6); // 1.5 MB

// ERROR CHECK UTILITY
void check_error(cudaError_t cudaerr) {
    if (cudaerr != cudaSuccess) {
        printf("FAILED WITH ERROR: \"%s\".\n", cudaGetErrorString(cudaerr));
        exit(-1);
    }   
}


// SETUP/CHECK UTILITIES

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
                memset((void*)&host[i], 0xff, 4);   
            }
        }
        case 2: {
            for (int i = 0; i < size/4; i++) {
                memset((void*)&host[i], 0x00, 4);   
            }
        }
        case 3: {
            for (int i = 0; i < size/4; i++) {
                memset((void*)&host[i], 0xf0, 4);   
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

int mem_check(int* host_pre, int* device, int size) {
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
	return 0;
}

// GPU KERNELS

__global__ void lxs(int* vals, size_t size, int stride, int toggles) {
    uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    int sum;
    for (int t = 0; t < toggles; t++) { 
        for (int i = tid; i < size/4; i += (i*stride)) {
            int n1 = vals[i];
            int n2 = vals[i+1];
            sum += n2 - n1; 
        }
    }   
    vals[0] = sum;    
}

// CPU CODE

int load_xmb_single() {
    printf("Input #blocks, #threads, cachesize multiplier, stride, toggles, repetitions, data pattern\n");
    int blocks, threads, stride, pattern, toggles, reps;
    double mult; 
    std::cin >> blocks;
    std::cin >> threads;
    std::cin >> mult;
    std::cin >> stride;
    std::cin >> toggles;
    std::cin >> reps;
    std::cin >> pattern;
    
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
	return 0;
}

int load_xmb_multi() {
	return 0;
}

int assoc() {
	return 0;
}

int hog() {
	return 0;
}


// RUN AND RERUN TESTS
int main(int argc, char** argv) {
	while (1) {
		printf("--------------------------\n");
		printf("GPU Rowhammer Test Mini-suite\n");
		printf("Options:\n\
				1. Run Test LoadXMB (Single Section)\n\
				2. Run Test LoadXMB (Multi Section)\n\
				3. Run Test Assoc\n\
				4. Run Test Hog\n\
				5. Test Descriptions\n\
				6. Set Cache Size\n\
				7. Quit\n");
		printf("--------------------------\n");
		int choice;
		std::cin >> choice;
		switch(choice) {
			case 1: {
				load_xmb_single();
				break;
			}
			case 2: {

				break;
			}
			case 3: {

				break;
			}
			case 4: {
			
				break;
			}
			case 5: {
				printf("--------------------------\n");
				printf("LoadXMB (Single Section)\n");
				printf("Allocate memory, size of some multiple of cachesize. Have threads loop through the memory\n\
						at some given stride (thread i accesses value i, i+(num threads * stride)...)\n");
				printf("--------------------------\n");
				printf("LoadXMB (Multiple Sections)\n");
				printf("Same as above, execpt n sections of the memory are allocated, and the thread grid is split\n\
						among the sections, looping through as before\n");
				printf("--------------------------\n");
				printf("Assoc\n");
				printf("Attempts to exploit associativity. Given some guessed level of associativity A and space S,\n\
						allocates n sections of (A*S) ints, sections spaced cachesize apart. Splits grid into n sections\n,\
						and these n sections repeatedly loop through the sections. Sort of scaled down version of LoadXMB\n\
						multiple, in an attempt to speed up memory accesses\n");
				printf("--------------------------\n");
				printf("Hog\n");
				printf("Test closest resembling google/original approach. A group of threads more or less does LoadXMB, while\n\
						another subset of the threads accesses a small set of addresses repeatedly\n");
				printf("--------------------------\n");	
				break;
			}
			case 6: {
				
				break;
			}
			case 7: {
				printf("Exiting\n");
				exit(0); 
			}
		}	
	}
}
