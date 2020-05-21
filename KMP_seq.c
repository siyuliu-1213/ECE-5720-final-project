/*
 * ECE 5720 Parallel Computing final project
 * Sequential substring matching
 * Shicong Li sl3295
 * Siyu Liu sl3282
 * Cornell University
 *
 * Compile : gcc -o KMP_seq KMP_seq.c
 * Run     : ./KMP_cuda
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#define BILLION 1E9L

// Compute LPS
void computeLPS(char* p, int* lps, int n);

int main(int argc, char *argv[]) {
    int n_p;
    int n_s;
    double time;
    struct timespec start, end;

    printf( "please put in the length of the haystack:\n" );
    scanf( "%d", &n_s );
    printf( "please put in the length of the needle:\n" );
    scanf( "%d", &n_p );

    // Read the target and pattern from a pre-written random-generated
    // composed of lowercase letter from a - z
    char* s = (char *) malloc(n_s * sizeof(char));
    char* p = (char *) malloc(n_p * sizeof(char));
    FILE * fptr = fopen( "data_5.txt" , "r");
    fgets(s, n_s + 1, fptr);
    fgets(p, n_p + 1, fptr);

    clock_gettime( CLOCK_MONOTONIC, &start );
    int* lps = (int *) malloc(n_p * sizeof(int));
    computeLPS(p, lps, n_p);

    // KMP algorithm
    int id_p = 0, id_s = 0;
    while(id_s < n_s) {
        if(s[id_s] == p[id_p]) {
            id_s++;
            id_p++;
        }

        if(id_p == n_p) {
            id_p = lps[id_p - 1];
        }

        else if(id_s < n_s && s[id_s] != p[id_p]) {
            if(id_p != 0) id_p = lps[id_p - 1];
            else id_s++;
        }
    }
    clock_gettime( CLOCK_MONOTONIC, &end );

    time = BILLION * ( end.tv_sec - start.tv_sec ) + end.tv_nsec - start.tv_nsec;
    printf("runs for %lf ns\n", time);
}

void computeLPS(char* p, int * lps, int n) {
    int len = 0;
    lps[0] = 0;

    int id = 1;
    while(id < n) {
        if(p[id] == p[len]) {
            len++;
            lps[id] = len;
            id++;
        }
        else {
            if(len != 0) len = lps[len - 1];
            else {
                lps[id] = 0;
                id++;
            }
        }
    }
}
