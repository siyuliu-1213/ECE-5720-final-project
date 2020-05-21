/*
 * ECE 5720 Parallel Computing final project
 * Substring matching with CUDA
 * Shicong Li sl3295
 * Siyu Liu sl3282
 * Cornell University
 *
 * Compile : /usr/local/cuda-10.1/bin/nvcc -arch=compute_52 -o KMP_cuda KMP_cuda.cu
 * Run     : ./KMP_cuda
 */

#include "cuda_profiler_api.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#define n_s 1e9
#define n_p 4
#define M 10000
#define N 8
#define BILLION 1E9L

__global__ void match(char *dev_s, char *dev_p, int *dev_lps, uint *dev_res_map)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    // calculate the start point and end point
    int start = i * n_s / (M * N);
    int end = (i + 1) * n_s / (M * N) + n_p - 1;
    // local variable for KMP matching
    int id_p = 0;
    int id_s = start;
    while(id_s < end) {
        if(dev_s[id_s] == dev_p[id_p]) {
            id_s++;
            id_p++;
        }

        if(id_p == n_p) {
            int idx = id_s - id_p;
            dev_res_map[idx / 32] |= 1 << (idx % 32);
            id_p = dev_lps[id_p - 1];
        }

        else if(id_s < end && dev_s[id_s] != dev_p[id_p]) {
            if(id_p != 0) id_p = dev_lps[id_p - 1];
            else id_s++;
        }
    }
}

void computeLPS(char* p, int* lps, int n) {
    // Initialization of lps array
    int len = 0;
    lps[0] = 0;

    int id = 1;
    while(id < n) {
        // record and move forward the pointer if character are identical
        if(p[id] == p[len]) {
            len++;
            lps[id] = len;
            id++;
        }

        // If not, move the  id pointer backward and compare again
        else {
            if(len != 0) len = lps[len - 1];
            else {
                lps[id] = 0;
                id++;
            }
        }
    }
}

int main() {
    char *s, *p, *dev_s, *dev_p;
    int *lps, *dev_lps;
    uint *res_map, *dev_res_map;

    s = (char *) malloc((n_s + n_p - 1) * sizeof(char));
    p = (char *) malloc(n_p * sizeof(char));
    lps = (int *) malloc(n_p * sizeof(int));
    res_map = (uint *) calloc((n_s/32), sizeof(uint));
    FILE * fptr = fopen( "../data_5.txt" , "r");
    fgets(s, n_s + 1, fptr);
    fgets(p, n_p + 1, fptr);
    fclose(fptr);
    for(int i = n_s; i < n_s + n_p - 1; i++) s[i] = p[1] + 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMalloc( (void**)&dev_s, (n_s + n_p - 1)*sizeof(char)); 
    cudaMalloc( (void**)&dev_p, n_p*sizeof(char));
    cudaMalloc( (void**)&dev_lps, n_p*sizeof(int)); 
    cudaMalloc( (void**)&dev_res_map, (n_s/32)*sizeof(uint)); 

    computeLPS(p, lps, n_p);
    cudaMemcpy(dev_s, s, (n_s + n_p - 1)*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p, p, n_p*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_lps, lps, n_p*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_res_map, res_map, (n_s/32)*sizeof(uint), cudaMemcpyHostToDevice);

    match<<<M, N>>>(dev_s, dev_p, dev_lps, dev_res_map);

    cudaMemcpy(res_map, dev_res_map, (n_s/32)*sizeof(uint), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total time is %lf\n", milliseconds);

    cudaFree(dev_s); cudaFree(dev_p); cudaFree(dev_lps); cudaFree(dev_res_map);
    free(s); free(p); free(lps); free(res_map);
}