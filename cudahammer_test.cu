#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(int** addrs, int addr_count, int toggles) {
	for (int i = 0; i < toggles; i++) {
		for (int a = 0; i < addr_count; i++) {
			asm("ld.global.cv r1 %0" : "r"(addrs[i]));
		}
	}
}

void hammer_attempt(int toggles, int addr_count, int mem_size, void* mem) {

	// declare timer
	GpuTimer timer;

	// declare array of addresses and fill it with addresses in our allocated memory
	int* addrs[addr_count];
	for (int i = 0; i < addr_count; i++) {
		int* new_addr = mem + (rand() % mem_size); 
		addrs[i] = new_addr; 
	}

	// start timer, launch kernel, stop timer
	timer.Start();
	kernel(addrs, addr_count);
	timer.Stop();

	// report time taken
	printf("%d memory accesses done on %d addresses in %g ms\n", toggles, addr_count, timer.Elapsed());

	// check for errors
	printf("checking for bit flips...\n");
	int flips = 0;
	int* pointer;
	int* mem_end = (int*) (mem + mem_size);
	for (pointer = (int*) mem; pointer < mem_end; pointer++) {
		int val = *pointer;
		if (val != (int) 0xff) {
			printf("found error!\n");
		}
	}
	printf("found %d errors\n", flips);
}


int main(int argc, char** argv) {
	
	// the amount of memory (in GB) to allocate for picking addresses from
	int num_gb = atoi(argv[1]); 
	int mem_size = (2 << 30) * num_gb;
	
	// for a particular set of chosen addresses, the number of times to access each one
	int toggles = atoi(argv[2]);

	// the number of addresses to be chosen for a hammering attempt
	// google test chooses 8, the original approach just uses 2 (has to be at least 2)
	int addr_count = atoi(argv[3]);

	// the number of times to choose a set of addresses
	// passing 0 makes the program continue to choose sets of addresses until stopped or until an error is found
	int iterations = atoi(argv[4]);

	// allocate a mem_size chunk of memory on the device
	int* mem; 
	cudaMalloc((void**)&mem, mem_size);

	// set all bits to 1
	cudaMemset(&mem, 0xff, (size_t) mem_size);

	// as mentioned above, either run iterations number of hammer attempts, or run until stopped
	if (iterations > 0) {
		for (int i = 0; i < iterations; i++) {
			hammer_attempt(toggles, addr_count, mem_size, mem);	
		}
	}	
	else {
		for (;;) {
			hammer_attempt(toggles, addr_count, mem_size, mem);	
		}
	}
}
