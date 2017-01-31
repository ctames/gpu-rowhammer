#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(int** addrs, int addr_count, int toggles) {
	// for (int i = 0; i < toggles; i++) {
		for (int a = 0; a < addr_count; a++) {
			asm("ld.param.cv.u32 %%rd1, [%0];" : "=l"(addrs[a]));
		}
	//}
}

void hammer_attempt(int toggles, int addr_count, const size_t mem_size, int* mem) {

	printf("entered hammer_attempt\n");
	// declare timer
	GpuTimer timer;

	// declare array of addresses and fill it with addresses in our allocated memory
	int* addrs[addr_count];
	for (int i = 0; i < addr_count; i++) {
		int* new_addr = (int*) mem + (rand() % mem_size); 
		addrs[i] = new_addr; 
	}

	// start timer, launch kernel, stop timer
	printf("starting kernel\n");
	timer.Start();
	for (int i = 0; i < toggles; i++) {
		printf("toggle %d\n", i);
		kernel<<<1,1>>>(addrs, addr_count, toggles);	
	}
	timer.Stop();

	// report time taken
	printf("%d memory accesses done on %d addresses in %g ms\n", toggles, addr_count, timer.Elapsed());

	// check for errors
	// copy device memory to host first
	printf("checking for bit flips...\n");
	printf("mem size sanity check: %d\n", mem_size);
	int* hostmem = (int*) malloc(mem_size);
	printf("hostmem address is %p\n", (void*) hostmem);
	printf("past host malloc\n");
	memset(hostmem, 0, mem_size);
	printf("past host memset\n");
	cudaError_t memcpyerr = cudaMemcpy(hostmem, mem, mem_size, cudaMemcpyDeviceToHost);	
	if (memcpyerr == cudaErrorInvalidValue) {
		printf("cudaMemcpy failed cudaErrorInvalidValue\n");		
		exit(-1);
	}
	if (memcpyerr == cudaErrorInvalidDevicePointer) {
		printf("cudaMemcpy failed cudaErrorInvalidDevicePointer\n");		
		exit(-1);
	}
	if (memcpyerr == cudaErrorInvalidMemcpyDirection) {
		printf("cudaMemcpy failed cudaErrorMemcpyDirection\n");		
		exit(-1);
	}
	printf("past the cudaMemcpyDeviceToHost\n");
	int flips = 0;
	int* pointer;
	int* mem_end = (int*) hostmem + mem_size;
	printf("mem_end address is %p\n", (void*) mem_end);
	for (pointer = hostmem; pointer < mem_end; pointer++) {
		int val = *pointer;
		//printf("past val\n");
		/*
		if (val != (int) 0xff) {
			flips++;
			//printf("found error!\n");
		}
		*/
		printf("%d\n", val);
	}
	printf("found %d errors\n", flips);
}


int main(int argc, char** argv) {
	 
	// amount of memory to allocate (1GB)
	const size_t mem_size = (1 << 30);
	
	// for a particular set of chosen addresses, the number of times to access each one
	int toggles = atoi(argv[1]);

	// the number of addresses to be chosen for a hammering attempt
	// google test chooses 8, the original approach just uses 2 (has to be at least 2)
	int addr_count = atoi(argv[2]);

	// the number of times to choose a set of addresses
	// passing 0 makes the program continue to choose sets of addresses until stopped or until an error is found
	int iterations = atoi(argv[3]);

	// allocate a mem_size chunk of memory on the device
	int* mem; 
	cudaError_t mallocerr = cudaMalloc((void**)&mem, mem_size);
	if (mallocerr == cudaErrorMemoryAllocation) {
		printf("cudaMalloc failed, exiting\n");
		exit(-1);
	}
	// set all bits to 1
	cudaError_t memseterr = cudaMemset(mem, 0xff, mem_size);
	if (memseterr == cudaErrorInvalidValue) {
		printf("cudaMemset failed cudaErrorInvalidValue\n");
		exit(-1);
	}
	if (memseterr == cudaErrorInvalidDevicePointer) {
		printf("cudaMemset failed cudaErrorInvalidDevicePointer\n");
		exit(-1);
	}
	

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
