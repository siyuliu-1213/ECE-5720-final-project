#include "mpi.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#define BILLION 1E9L

void computeLPS(char* p, int* lps, int n);
void match(char* s, char* p, int* lps, int n_s, int n_p, int start);
void print(char* s, int rank, int len);

int main(int argc, char *argv[]) {
    char* s;
    char* p;
    long n_s = 1e6;
    int n_p = 7;
    int* lps;
    double start_time, end_time, time;
    int numtasks, rank;

    // MPI initialization
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Use first process to get input and 
    // broadcast to other process
    p = (char *) malloc(n_p * sizeof(char));
    lps = (int *) malloc(n_p * sizeof(int));

    if(rank == 0) {
        // Get input
        // printf( "please put in the length of the haystack:\n" );
        // scanf( "%ld", &n_s );
        // printf( "please put in the length of the needle:\n" );
        // scanf( "%d", &n_p );

        // Initialization
        s = (char *) malloc(n_s * sizeof(char));
        FILE * fptr = fopen( "data_5.txt" , "r");
        fgets(s, n_s + 1, fptr);
        fgets(p, n_p + 1, fptr);
        fclose(fptr);
    }
    start_time = MPI_Wtime();
    if(rank == 0) computeLPS(p, lps, n_p);
    // Broadcast n_s and pattern to all the processes
    MPI_Bcast(p, n_p, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(lps, n_p, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Local area of target string
    int start = (n_s * rank) / numtasks;
    int end = (n_s * (rank + 1)) / numtasks;
    
    if(rank == 0) {
        // Distribute target string among all the processes
        MPI_Request reqs[numtasks - 1];   // required variable for non-blocking calls
        MPI_Status stats[numtasks - 1];   // required variable for Waitall routine

        for(int i = 1; i < numtasks; i++) {
            // calculate the size certain process needs to address
            int start_p = (n_s * i) / numtasks;
            int end_p = (n_s * (i + 1)) / numtasks;

            // for the last process, only need to process to the end
            if(i == numtasks - 1) MPI_Isend(&s[start_p], end_p - start_p, MPI_CHAR, i, i, MPI_COMM_WORLD, &reqs[i - 1]);
            else MPI_Isend(&s[start_p], end_p - start_p + n_p - 1, MPI_CHAR, i, i, MPI_COMM_WORLD, &reqs[i - 1]);
        }

        // print(p, rank, n_p);
        // print(s, rank, end - start + n_p - 1);
        match(s, p, lps, end - start + n_p - 1, n_p, start);
        MPI_Waitall(numtasks - 1, reqs, stats);
        free(s);
    }
    else {
        MPI_Request reqs;   // required variable for non-blocking calls
        MPI_Status stats;   // required variable for Waitall routine
        char* s_local;
        if(rank == numtasks - 1) s_local = (char*) malloc((end - start) * sizeof(char));
        else s_local = (char*) malloc((end - start + n_p - 1) * sizeof(char));

        if(rank == numtasks - 1) {
            MPI_Irecv(s_local, end - start, MPI_CHAR, 0, rank, MPI_COMM_WORLD, &reqs);
            MPI_Wait(&reqs, &stats);
            // print(p, rank, n_p);
            // print(s_local, rank, end - start);
            match(s_local, p, lps, end - start, n_p, start);
        }
        else {
            MPI_Irecv(s_local, end - start + n_p - 1, MPI_CHAR, 0, rank, MPI_COMM_WORLD, &reqs);
            MPI_Wait(&reqs, &stats);
            // print(p, rank, n_p);
            // print(s_local, rank, end - start + n_p - 1);
            match(s_local, p, lps, end - start + n_p - 1, n_p, start);
        }
        free(s_local);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    if(rank == 0) {
        time = (end_time - start_time) * BILLION;
        printf("runs for %lf ns\n", time);
    }
    free(lps);
    free(p);
    MPI_Finalize();
}

void computeLPS(char* p, int* lps, int n) {
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

void match(char* s, char* p, int* lps, int n_s, int n_p, int start) {
    int id_p = 0;
    int id_s = 0;
    // Go through every character to find the match
    while(id_s < n_s) {
        // If characters are matched, move both pointer forward
        if(s[id_s] == p[id_p]) {
            id_s++;
            id_p++;
        }

        // If substring match is found, save it in a map
        if(id_p == n_p) {
            printf("Found matching at index %d \n", id_s - id_p + start);
            id_p = lps[id_p - 1];
        }

        // In terms of mismatch, move only pattern pointer backward
        else if(id_s < n_s && s[id_s] != p[id_p]) {
            if(id_p != 0) id_p = lps[id_p - 1];
            else id_s++;
        }
    }
}

void print(char* s, int rank, int len) {
    printf("Process %d is addressing: ", rank);
    for(int i = 0; i < len; i++) printf("%c", s[i]);
    printf("\n");
}