/* Gaussian elimination code.
 * 
 * Author: Naga Kandasamy
 * Date modified: April 26, 2023
 *
 * Student names(s): Noah Robinson, Jack Pinkstone
 * Date: 5/7/2023
 *
 * Compile as follows: 
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -std=c99 -O3 -Wall -lpthread -lm
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50

/* Data structure defining what to pass to each worker thread */
typedef struct thread_data_s {
    int tid;                        /* The thread ID */
    int num_threads;                /* Number of threads in the worker pool */
    int chunk_size;
    int offset;
    int num_rows;
    float *elements;
    pthread_barrier_t *barrier;
} thread_data_t;

/* Function prototypes */
extern int compute_gold(float *, int);
Matrix allocate_matrix(int, int, int);
void gauss_eliminate_using_pthreads(Matrix, int);
void *gauss_worker(void *);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, int, float);

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s matrix-size\n", argv[0]);
        fprintf(stderr, "matrix-size: width and height of the square matrix\n");
        exit(EXIT_FAILURE);
    }

    int matrix_size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    

    Matrix A;			                                            /* Input matrix */
    Matrix U_reference;		                                        /* Upper triangular matrix computed by reference code */
    Matrix U_mt;			                                        /* Upper triangular matrix computed by pthreads */

    fprintf(stderr, "Generating input matrices\n");
    srand(time (NULL));                                             /* Seed random number generator */
    A = allocate_matrix(matrix_size, matrix_size, 1);               /* Allocate and populate random square matrix */
    U_reference = allocate_matrix (matrix_size, matrix_size, 0);    /* Allocate space for reference result */
    U_mt = allocate_matrix (matrix_size, matrix_size, 0);           /* Allocate space for multi-threaded result */

    /* Copy contents A matrix into U matrices */
    int i, j;
    for (i = 0; i < A.num_rows; i++) {
        for (j = 0; j < A.num_rows; j++) {
            U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
            U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
        }
    }

    fprintf(stderr, "\nPerforming gaussian elimination using reference code\n");
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    
    int status = compute_gold(U_reference.elements, A.num_rows);
  
    gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

    if (status < 0) {
        fprintf(stderr, "Failed to convert given matrix to upper triangular. Try again.\n");
        exit(EXIT_FAILURE);
    }
  
    status = perform_simple_check(U_reference);	/* Check that principal diagonal elements are 1 */ 
    if (status < 0) {
        fprintf(stderr, "Upper triangular matrix is incorrect. Exiting.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Single-threaded Gaussian elimination was successful.\n");
  
    /* FIXME: Perform Gaussian elimination using pthreads. 
     * The resulting upper triangular matrix should be returned in U_mt */
    fprintf(stderr, "\nPerforming gaussian elimination using pthreads\n");
    
    struct timeval start1, stop1;

    gettimeofday(&start1, NULL);
    gauss_eliminate_using_pthreads(U_mt, num_threads);
    gettimeofday(&stop1, NULL);

    fprintf(stderr, "CPU run time = %0.2f s\n", (float)(stop1.tv_sec - start1.tv_sec\
                + (stop1.tv_usec - start1.tv_usec) / (float)1000000));

    /* Check if pthread result matches reference solution within specified tolerance */
    fprintf(stderr, "\nChecking results\n");
    int size = matrix_size * matrix_size;
    int res = check_results(U_reference.elements, U_mt.elements, size, 1e-6);
    fprintf(stderr, "TEST %s\n", (0 == res) ? "PASSED" : "FAILED");

    /* Free memory allocated for matrices */
    free(A.elements);
    free(U_reference.elements);
    free(U_mt.elements);

    exit(EXIT_SUCCESS);
}


/* FIXME: Write code to perform gaussian elimination using pthreads */
void gauss_eliminate_using_pthreads(Matrix U, int num_threads)
{
    /* FIXME: Complete this function */
    pthread_t *thread_id = (pthread_t *)malloc (num_threads * sizeof(pthread_t)); /* Data structure to store the thread IDs */
    pthread_attr_t attributes;      /* Thread attributes */
    pthread_attr_init(&attributes); /* Initialize thread attributes to default values */
		  
    /* Fork point: allocate memory on heap for required data structures and create worker threads */
    int i;
    int chunk_size = (int)floor((float)U.num_rows/(float) num_threads); /* Compute the chunk size */

    thread_data_t *thread_data = (thread_data_t *) malloc(sizeof(thread_data_t) * num_threads);
    
    pthread_barrier_t *barrier = (pthread_barrier_t *)malloc(sizeof(pthread_barrier_t *));
    pthread_barrier_init(barrier,NULL,num_threads);

    for (i = 0; i < num_threads; i++) {
        thread_data[i].tid = i; 
        thread_data[i].num_threads = num_threads;
        thread_data[i].offset = i * chunk_size; 
        thread_data[i].chunk_size = chunk_size;
        thread_data[i].barrier = barrier;
        thread_data[i].num_rows = U.num_rows;
        thread_data[i].elements = U.elements;
    }

    for (i = 0; i < num_threads; i++)
        pthread_create(&thread_id[i], &attributes, gauss_worker, (void *)&thread_data[i]);
					 
    /* Join point: wait for the workers to finish */
    for (i = 0; i < num_threads; i++)
        pthread_join(thread_id[i], NULL);
		
    /* Free dynamically allocated data structures */
    free((void *)thread_data);

    return;
}

void *gauss_worker(void *args)
{
    /* Typecast argument as a pointer to the thread_data_t structure */
    thread_data_t *thread_data = (thread_data_t *)args; 		  

    int i, j, k;
    int stride = thread_data->num_threads;

    for (k = 0; k < thread_data->num_rows; k++) {
                    
        for (j = (k + thread_data->tid + 1); j < thread_data->num_rows; j = j + stride) 
        {
            /* Division step */
            thread_data->elements[thread_data->num_rows * k + j] = (float) (thread_data->elements[thread_data->num_rows * k + j] / thread_data->elements[thread_data->num_rows * k + k]); 
        }
        pthread_barrier_wait(thread_data->barrier);

        for (i = (thread_data->tid + k + 1); i < thread_data->num_rows; i = i + stride) 
        {
            /* Elimination step */
            for (j = (k + 1); j < thread_data->num_rows; j++) {
                thread_data->elements[thread_data->num_rows * i + j] -= (thread_data->elements[thread_data->num_rows * i + k] * thread_data->elements[thread_data->num_rows * k + j]);
            }
            thread_data->elements[thread_data->num_rows * i + k] = 0;
        }
        pthread_barrier_wait(thread_data->barrier);
    }

    /* Set the principal diagonal entry in U to 1 */ 
    for (k = 0 + thread_data->tid; k < thread_data->num_rows; k = k + stride) {
         thread_data->elements[thread_data->num_rows * k + k] = 1; 
    }

    pthread_exit ((void *)0);
    pthread_exit(NULL);

    //pivot logic
    // int   i,k,m,rowx;
    // double xfac, temp, temp1, amax;

    // amax = (double) fabs(a[j][j]);
    // m = j;
    // for (i=j+1; i<n; i++){   /* Find the row with largest pivot */
    //     xfac = (double) fabs(a[i][j]);
    //     if(xfac > amax) {amax = xfac; m=i;}
    // }

    // if(m != j) {  /* Row interchanges */
    //     rowx = rowx+1;
    //     temp1 = b[j];
    //     b[j]  = b[m];
    //     b[m]  = temp1;
    //     for(k=j; k<n; k++) {
    //         temp = a[j][k];
    //         a[j][k] = a[m][k];
    //         a[m][k] = temp;
    //     }
    // }
}

/* Check if results generated by single threaded and multi threaded versions match within tolerance */
int check_results(float *A, float *B, int size, float tolerance)
{
    int i;
    for (i = 0; i < size; i++)
        if(fabsf(A[i] - B[i]) > tolerance)
            return -1;
    return 0;
}

/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization. 
*/
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;
    Matrix M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
    M.elements = (float *)malloc(size * sizeof(float));
  
    for (i = 0; i < size; i++) {
        if (init == 0)
            M.elements[i] = 0;
        else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
    }
  
    return M;
}

/* Return a random floating-point number between [min, max] */ 
float get_random_number(int min, int max)
{
    return (float)floor((double)(min + (max - min + 1) * ((float)rand() / (float)RAND_MAX)));
}

/* Perform simple check on upper triangular matrix if the principal diagonal elements are 1 */
int perform_simple_check(const Matrix M)
{
    int i;
    for (i = 0; i < M.num_rows; i++)
        if ((fabs(M.elements[M.num_rows * i + i] - 1.0)) > 1e-6)
            return -1;
  
    return 0;
}