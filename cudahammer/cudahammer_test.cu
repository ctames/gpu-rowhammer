#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h> 

__global__ void kernel(int** addrs, int addr_count, int toggles, int* mem, int mem_size) {
	int sum = 0;
	for (int i = 0; i < toggles; i++) {
		for (int a = 0; a < addr_count; a++) {
			int x = *(addrs[a]);
			sum += x;
			//asm("ld.param.cv.u32 %0, [%1];" : "=r"(x) : "l"(addrs[a]));
			//asm("ld.param.u32 %0, [%1];" : "=r"(x) : "l"(addrs[a]));
			//printf("x: %d\n", x);
		}
	}
}

void check_error(cudaError_t cudaerr) {
	if (cudaerr != cudaSuccess) {
		printf("FAILED WITH ERROR: \"%s\".\n", cudaGetErrorString(cudaerr));
		exit(-1);
	}
}

void hammer_attempt(int toggles, int addr_count, const size_t mem_size, int* mem) {
	// SETUP
	printf("entered hammer_attempt\n");
	int num_ints = mem_size / 4;
	int cudaerr;	

	// declare timer
	GpuTimer timer;
	
	// cudaMalloc for an array of addresses
	int** addrs;
	cudaerr = cudaMalloc((void**)&addrs, sizeof(int*) * addr_count);
	check_error(cudaerr);

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
	cudaerr = cudaMemcpy(addrs, host_addrs, sizeof(int*) * addr_count, cudaMemcpyHostToDevice);	
	check_error(cudaerr);
	printf("past the addr cudaMemcpy\n");
		
	// PERFORM THE HAMMERING MEMORY ACCESSES
	// start timer, launch kernel, stop timer
	printf("starting kernel\n");
	timer.Start();
	kernel<<<1,1>>>(addrs, addr_count, toggles, mem, mem_size);	
	timer.Stop();
	cudaerr = cudaDeviceSynchronize();
	check_error(cudaerr);	

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
	cudaerr = cudaMemcpy(hostmem, mem, mem_size, cudaMemcpyDeviceToHost);	
	check_error(cudaerr);
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

struct test_opts {
	int iterations;
	int toggles;
	int addr_count;
	int numvals; 
	int blocks;
	int threads;
	int data_pattern;
	int logging;
	char byte_value;
}

void usage(char** argv) {
	printf("Cudahammer test usage:\n
%s [-i iterations] [-t toggles] [-a addr_count] [-b blocks] [-h threads] [-n numvals] [-p data_pattern] [-v byte_value] [-l]\n
OPTION		DEFAULT		DESCRIPTION\n			
-i			10			(int) The number of times to pick a set of addresses to hammer (0 means infinite).\n
-t			500000		(int) The number of times to toggle each address.\n
-a			8			(int) The number of addresses to pick for a single iteration of the hammer attempt.\n
-b			1			(int) The number of blocks for the test kernel launch.\n
-h			1			(int) The number of threads for the test kernel launch.\n
-n			268435456	(int) The number of 32-bit ints to allocate memory for (default is 1GB worth).\n
-p			1			(int) The data pattern to use - check the README for more explanation.\n
-v			0xff		(char) The single byte value to memset the allocated memory with.\n
-l			off			Disply logging messages during test operation.\n
-u			off			Display this usage message and exit.\n", argv[0]);
}

int main(int argc, char** argv) {
	
	// process command line options	
	const char *opts_string = ""; 
	struct test_opts opts = {10, 500000, 8, 268435456, 1, 1, 1, 0, 0xff};	
	int flag;
	while((flag = getopt(argc, argv, opts_string) != -1) { 
		switch(flag) {
			case 'i':
				opts.iterations = atoi(optarg);
				break;
			case 't':
				opts.toggles = atoi(optarg);
				break;
			case 'a':
				opts.addr_count = atoi(optarg);
				break;
			case 'b':
				opts.blocks = atoi(optarg);
				break;
			case 'h':
				opts.threads = atoi(optarg);
				break;
			case 'n':
				opts.numvals = atoi(optarg);
				break;
			case 'p':
				opts.data_pattern = atoi(optarg);
				break;
			case 'v':
				opts.byte_value = atoc(optarg);
				break;
			case 'l':
				opts.logging = 1;
				break;
			case 'u':
				usage(argv);
				exit(-1);
			case '?':
				usage(argv);
				exit(-1);
			default:
				usage(argv);
				exit(-1);
		}
	}
 
	// allocate a chunk of memory on the device
	int* mem; 
	int cudaerr;
	cudaerr = cudaMalloc((void**)&mem, sizeof(int)*numvals);
	check_error(cudaerr);

	// write data pattern to bits
	cudaerr = cudaMemset(mem, 0xff, mem_size);
	check_error(cudaerr);
	
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
