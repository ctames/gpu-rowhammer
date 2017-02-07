#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void kernel(int** addrs, int addr_count, int toggles, int* mem, int mem_size) {
	/*
	for (int i = 0; i < toggles; i++) {
		for (int a = 0; a < addr_count; a++) {
			asm("ld.param.cv.u32 %%rd1, [%0];" : : "l"(addrs[a]));
		}
	}
	*/
	
	for (int i = 0; i < toggles; i++) {
		for (int a = 0; a < addr_count; a++) {
			int x;
			asm("ld.param.cv.u32 %0, [%1];" : "=r"(x) : "l"(addrs[a]));
			printf("x: %d\n", x);
		}
	}
}

void hammer_attempt(int toggles, int addr_count, const size_t mem_size, int* mem) {
	// SETUP
	printf("entered hammer_attempt\n");
	int num_ints = mem_size / 4;
	
	// declare timer
	GpuTimer timer;
	
	// cudaMalloc for an array of addresses
	int** addrs;
	int mallocerr = cudaMalloc((void**)&addrs, sizeof(int*) * addr_count);
	if (mallocerr == cudaErrorMemoryAllocation) {
		printf("cudaMalloc failed, exiting\n");
		exit(-1);
	}

	// print out address range of device memory for sanity checking randomly chosen pointers
	printf("mem address: %p\n", (void*)mem);
	int* device_mem_end = (int*) mem + num_ints;
	printf("device_mem_end address: %p\n", (void*)device_mem_end);

	// malloc host array of addresses, fill it, memcpy to device array
	int** host_addrs = (int**) malloc(sizeof(int*) * addr_count); 
	srand(time(NULL));
	for (int i = 0; i < addr_count; i++) {
		int* new_addr = (int*) mem + (rand() % num_ints); 
		host_addrs[i] = new_addr;
		printf("address: %p\n", (void*)host_addrs[i]);
	}
	cudaError_t memcpyerr = cudaMemcpy(addrs, host_addrs, sizeof(int*) * addr_count, cudaMemcpyHostToDevice);	
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
	printf("past the addr cudaMemcpy\n");
		
	// PERFORM THE HAMMERING MEMORY ACCESSES
	// start timer, launch kernel, stop timer
	printf("starting kernel\n");
	timer.Start();
	kernel<<<1,1>>>(addrs, addr_count, toggles, mem, mem_size);	
	timer.Stop();
	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess)
		printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
	
	// report time taken 
	printf("%d memory accesses done on %d addresses in %g ms\n", toggles, addr_count, timer.Elapsed());

	// CHECKING FOR FLIPS SECTION
	printf("starting sanity checks and memory copy to host...\n");
	
	// allocate memory on host
	int* hostmem = (int*) malloc(mem_size);
	printf("mem size: %d\n", mem_size);
	printf("hostmem address is %p\n", (void*) hostmem);
	printf("past host malloc\n");
	
	// blank out hostmem to start with
	memset(hostmem, 0, mem_size);
	printf("past host memset\n");
	
	// copy device memory to host
	memcpyerr = cudaMemcpy(hostmem, mem, mem_size, cudaMemcpyDeviceToHost);	
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
	
	// calculate end of host mem address
	int* mem_end = (int*) hostmem + num_ints;
	printf("host mem_end address is %p\n", (void*) mem_end);

	// time to actually check for bit flips
	printf("checking for bit flips...");
	int flips = 0;
	int val = 0;
	int* pointer;
	for (pointer = hostmem; pointer < mem_end; pointer++) {
		val = *pointer;
		if (val != -1) {
			flips++;
			printf("found error!\n");
		}
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
