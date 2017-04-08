#include <iostream>
#include <hammertests.h>

int size_t cachesize = 1.5 * (1 << 6) // 1.5 MB

int main(int argc, char** argv) {
	while (1) {
		printf("--------------------------\n");
		printf("GPU Rowhammer Test Mini-suite\n");
		printf("Options:\n
				1. Run Test LoadXMB (Single Section)\n
				2. Run Test LoadXMB (Multi Section)\n
				3. Run Test Assoc\n
				4. Run Test Hog\n
				5. Test Descriptions\n
				6. Set Cache Size\n
				7. Quit\n");
		printf("--------------------------\n");
		int choice;
		cin >> choice;
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
				printf("Allocate memory, size of some multiple of cachesize. Have threads loop through the memory\n
						at some given stride (thread i accesses value i, i+(num threads * stride)...)\n");
				printf("--------------------------\n");
				printf("LoadXMB (Multiple Sections)\n");
				printf("Same as above, execpt n sections of the memory are allocated, and the thread grid is split\n
						among the sections, looping through as before\n");
				printf("--------------------------\n");
				printf("Assoc\n");
				printf("Attempts to exploit associativity. Given some guessed level of associativity A and space S,\n
						allocates n sections of (A*S) ints, sections spaced cachesize apart. Splits grid into n sections\n,
						and these n sections repeatedly loop through the sections. Sort of scaled down version of LoadXMB\n
						multiple, in an attempt to speed up memory accesses\n");
				printf("--------------------------\n");
				printf("Hog\n");
				printf("Test closest resembling google/original approach. A group of threads more or less does LoadXMB, while\n
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
