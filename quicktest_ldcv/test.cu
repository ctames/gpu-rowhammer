#include <stdio.h>

__global__ void test(int* dataD, int* sumD) {
	for (int i = 0; i < 1000000; i++) {
		int x = dataD[0];
		int y = dataD[1];
		int z = dataD[2];
		int sum = x+y+z;
		*sumD += sum;
	}
}

int main() {
	int* dataH = (int*)malloc(sizeof(int)*10);
	for (int i = 0; i < 10; i++) {
		dataH[i] = i; 
	}
	int* dataD;
	cudaMalloc((void**)&dataD, sizeof(int)*10);
	cudaMemcpy(dataD, dataH, sizeof(int)*10, cudaMemcpyHostToDevice);
	int* sumH = (int*)malloc(sizeof(int));
	int* sumD; 
	cudaMalloc((void**)&sumD, sizeof(int));
	test<<<1,1>>>(dataD, sumD);
	cudaMemcpy(sumH, sumD, sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d\n", *sumH);
}
