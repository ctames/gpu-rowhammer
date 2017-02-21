/* -*- mode: c++ -*- */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <stdlib.h>
#include <unistd.h>
#include "snfile.h"
#include <cassert>

struct prog_opts {
  int sort;
  int tbsize;
  int blocks;
  int nwlfiles;
  size_t nodes;
  char **wlfiles;
};

__global__ void test(int *arr, int N, int *offsets) {
  uint tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint nthreads = blockDim.x * gridDim.x;
  int sum = 0;
  clock_t s, e;
  int x = 0;
  s = clock64();
  
  for(int i = tid; i < N; i+= nthreads) {
    int node = offsets[i];
    int n1 = arr[node];
    int n2 = arr[node+1];
    sum += n2 - n1;
    x++;
  }

  e = clock64();

  if(tid == 0) {
    arr[tid] = sum + (e - s) / x; 
  }
}

void check_cuda(cudaError_t r) {
  if(r != cudaSuccess) {
    fprintf(stderr, "Error: %d: %s\n", r, cudaGetErrorString(r));
    exit(1);
  }
}

int comparint(const void *x, const void *y) {
  int xx = *(const int *) x;
  int yy = *(const int *) y;

  return xx - yy;
}

int * read_offsets_snappy(const char *of, size_t *N, int sort) {
  SNAPPY_FILE f;

  f = snopen(of, "r");

  if(!f) {
    fprintf(stderr, "Unable to open '%s'\n", of);
    exit(1);
  }

  if(snread(f, N, sizeof(*N) * 1) != sizeof(*N) * 1) {
    fprintf(stderr, "Unable to read length\n");
    exit(1);
  }
    
  int *offsets;

  offsets = (int *) calloc(*N, sizeof(int));
  if(!offsets) {
    fprintf(stderr, "Unable to allocate memory for offsets\n");
    exit(1);
  }

  if(snread(f, offsets, sizeof(int) * (*N)) != sizeof(int) * (*N)) {
    fprintf(stderr, "Unable to read offsets\n");
    exit(1);
  }

  snclose(f);

  fprintf(stderr, "Read %llu offsets from compressed file\n", *N);

  if(sort) {
    qsort(offsets, *N, sizeof(int), comparint);

    if(*N < 10) {
      for(int i = 0; i < *N; i++) {
	printf("%d %d\n", offsets[i]);
      }
    }

  }

  return offsets;
}

int * read_offsets(const char *of, size_t *N, int sort) {
  FILE *f = fopen(of, "r");
  if(!f) {
    fprintf(stderr, "Unable to open '%s'\n", of);
    exit(1);
  }

  if(fread(N, sizeof(*N), 1, f) != 1) {
    fprintf(stderr, "Unable to read length\n");
    exit(1);
  }
    
  int *offsets;

  offsets = (int *) calloc(*N, sizeof(int));
  if(!offsets) {
    fprintf(stderr, "Unable to allocate memory for offsets\n");
    exit(1);
  }

  if(fread(offsets, sizeof(int), *N, f) != *N) {
    fprintf(stderr, "Unable to read offsets\n");
    exit(1);
  }

  fclose(f);

  fprintf(stderr, "Read %llu offsets\n", *N);

  if(sort) {
    qsort(offsets, *N, sizeof(int), comparint);

    if(*N < 10) {
      for(int i = 0; i < *N; i++) {
	printf("%d %d\n", offsets[i]);
      }
    }

  }

  return offsets;
}

void usage(int argc, char *argv[]) {
  fprintf(stderr, "Usage: %s [-s] [-t tbsize] [-b blocks] nodes worklist-files\n", argv[0]);
}

void process_opts(int argc, char *argv[], struct prog_opts *opts) {
  const char *indirect_opts = "t:b:s";
  int c;

  while((c = getopt(argc, argv, indirect_opts)) != -1) {
    switch(c) {
    case 's':
      opts->sort = 1;
      break;
    case 't':
      opts->tbsize = atoi(optarg);
      break;
    case 'b':
      opts->blocks = atoi(optarg);
      break;
    case '?':
      usage(argc, argv);
      exit(EXIT_FAILURE);
    default:
      assert(false);
    }    
  }  

  if(optind + 2 <= argc) {
    opts->nodes = atoi(argv[optind]);
    assert(opts->nodes > 0);
    opts->wlfiles = &argv[optind + 1];
    opts->nwlfiles = argc - (optind + 1);
  } else {
    usage(argc, argv);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
  struct prog_opts po;
  po.sort = 0;
  po.tbsize = 1;
  po.blocks = 1;

  /* if(4 <= argc) sort = atoi(argv[3]);
  if(5 <= argc) tbsize = atoi(argv[4]);
  if(6 <= argc) blocks = atoi(argv[5]);*/

  process_opts(argc, argv, &po);

  printf("nodes: %d sort=%d tbsize=%d blocks=%d\n", po.nodes, po.sort, po.tbsize, po.blocks);

  size_t arraysize = po.nodes * sizeof(int);
  size_t N;

  int *arrH, *arrG;
  arrH = (int *) calloc(po.nodes, sizeof(int));
    
  for(int i = 0; i < po.nodes; i++) {
    arrH[i] = random();
  }

  check_cuda(cudaMalloc(&arrG, arraysize));

  printf("arrG: %p\n", arrG);
  check_cuda(cudaMemcpy(arrG, arrH, arraysize, cudaMemcpyHostToDevice));
  
  for(int i = 0; i < po.nwlfiles; i++) {
    printf("reading %s (%d)\n", po.wlfiles[i], i);

    int *offsets = read_offsets_snappy(po.wlfiles[i], &N, po.sort);
    int *offsets_G;

    check_cuda(cudaMalloc(&offsets_G, N * sizeof(int)));  
    check_cuda(cudaMemcpy(offsets_G, offsets, N * sizeof(int), cudaMemcpyHostToDevice));

    test<<<po.blocks, po.tbsize>>>(arrG, N, offsets_G);
    //check_cuda(cudaMemcpy(arrH, arrG, arraysize, cudaMemcpyDeviceToHost));
    //printf("cycles per access: %d (%d %d)\n", arrH[0], nodes, N);

    check_cuda(cudaDeviceSynchronize());
    check_cuda(cudaFree(offsets_G));
    free(offsets);
  }
}
