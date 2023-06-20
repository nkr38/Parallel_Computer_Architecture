/* Host code that implements a  separable convolution filter of a 
 * 2D signal with a gaussian kernel.
 * 
 * Author: Naga Kandasamy
 * Date modified: May 26, 2020
 * Student/team: Noah Robinson and Jack Pinkstone
 * Date: 6/12/23
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

extern "C" void compute_gold(float *, float *, int, int, int);
extern "C" float *create_kernel(float, int);
void print_kernel(float *, int);
void print_matrix(float *, int, int);

/* Width of convolution kernel */
#define HALF_WIDTH 8
#define COEFF 10
#define THREAD_BLOCK_SIZE 4

/* Optimize using constant memory */
__constant__ float kernel_optimized[HALF_WIDTH * 2 + 1];

/* Uncomment line below to spit out debug information */
// #define DEBUG

/* Include device code */
#include "separable_convolution_kernel.cu"

/* FIXME: Edit this function to compute the convolution on the device.*/
void compute_on_device(float *gpu_result, float *matrix_c, float *kernel, int num_cols, int num_rows, int half_width, float *result_with_optimization)
{
    
    int num_elements = num_rows * num_cols, width = 2 * HALF_WIDTH + 1;
    float *device_result, *device_input, *device_kernel, *device_input_opt; 
    size_t mem_size = num_elements * sizeof(float);

    /* Allocate memory */
    cudaMalloc((void**)&device_result, mem_size);
    cudaMalloc((void**)&device_input, mem_size);
    cudaMalloc((void**)&device_input_opt, mem_size);
    cudaMalloc((void**)&device_kernel, width * sizeof(float));

    /* Copy memory from host to device */
    cudaMemcpy(device_input, matrix_c, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_input_opt, matrix_c, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_kernel, kernel, width * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    int dim1 = num_cols/THREAD_BLOCK_SIZE;
    int dim2 = num_rows/THREAD_BLOCK_SIZE;
    dim3 grid(dim1, dim2);

    /* Timing variables */
    struct timeval start, stop;
    struct timeval start2, stop2;

    gettimeofday(&start, NULL);
    convolve_rows_kernel_naive<<< grid, threads >>>(device_result, device_input, device_kernel, num_cols, num_rows, half_width);
    /* Sync necessary */
    cudaDeviceSynchronize();
    convolve_columns_kernel_naive<<< grid, threads >>>(device_input, device_result, device_kernel, num_cols, num_rows, half_width);
    gettimeofday(&stop, NULL);

    float exec_time1 = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000);
    printf ("[Naive] = %13fs\n", exec_time1);

    /* Copy result out of device */
    cudaMemcpy(gpu_result, device_input, mem_size, cudaMemcpyDeviceToHost);

    gettimeofday(&start2, NULL);
    cudaMemcpyToSymbol(kernel_optimized, kernel, width * sizeof(float));
    convolve_rows_kernel_optimized<<< grid, threads >>>(device_result, device_input_opt, num_cols, num_rows, half_width);
    /* Sync necessary */
    cudaDeviceSynchronize();
    convolve_columns_kernel_optimized<<< grid, threads >>>(device_input_opt, device_result, num_cols, num_rows, half_width);
    gettimeofday(&stop2, NULL);

    float exec_time2 = (float)(stop2.tv_sec - start2.tv_sec + (stop2.tv_usec - start2.tv_usec) / (float)1000000);
    printf ("[Optim] = %13fs\n", exec_time2);

    /* Copy result out of device */
    cudaMemcpy(result_with_optimization, device_input, mem_size, cudaMemcpyDeviceToHost);

    /* Free pointers */
    cudaFree(device_result);
    cudaFree(device_input);
    cudaFree(device_kernel);
    cudaFree(device_input_opt);

    return;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("Usage: %s num-rows num-columns\n", argv[0]);
        printf("num-rows: height of the matrix\n");
        printf("num-columns: width of the matrix\n");
        exit(EXIT_FAILURE);
    }

    int num_rows = atoi(argv[1]);
    int num_cols = atoi(argv[2]);

    /* Create input matrix */
    int num_elements = num_rows * num_cols;
    printf("Creating input matrix of %d x %d\n", num_rows, num_cols);
    float *matrix_a = (float *)malloc(sizeof(float) * num_elements);
    float *matrix_c = (float *)malloc(sizeof(float) * num_elements);
	
    srand(time(NULL));
    int i;
    for (i = 0; i < num_elements; i++) {
        matrix_a[i] = rand()/(float)RAND_MAX;			 
        matrix_c[i] = matrix_a[i]; /* Copy contents of matrix_a into matrix_c */
    }
	 
	/* Create Gaussian kernel */	  
    float *gaussian_kernel = create_kernel((float)COEFF, HALF_WIDTH);	
#ifdef DEBUG
    print_kernel(gaussian_kernel, HALF_WIDTH); 
#endif
	  
    struct timeval start, stop;
    //printf("\nConvolving the matrix on the CPU\n");	  
    gettimeofday (&start, NULL);
    compute_gold(matrix_a, gaussian_kernel, num_cols,\
                  num_rows, HALF_WIDTH);
    gettimeofday (&stop, NULL);
    float exec_time = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000);
    printf("\n[Gold] = %14fs\n", exec_time); 

#ifdef DEBUG	 
    print_matrix(matrix_a, num_cols, num_rows);
#endif
  
    float *gpu_result = (float *)malloc(sizeof(float) * num_elements);
    float *result_with_optimization = (float *)malloc(sizeof(float) * num_elements);

    //printf("\nConvolving matrix on the GPU\n");
    compute_on_device(gpu_result, matrix_c, gaussian_kernel, num_cols,\
                       num_rows, HALF_WIDTH, result_with_optimization);
       
    printf("\nComparing naive CPU and GPU results\n");
    float sum_delta = 0, sum_ref = 0;
    for (i = 0; i < num_elements; i++) {
        sum_delta += fabsf(matrix_a[i] - gpu_result[i]);
        sum_ref   += fabsf(matrix_a[i]);
    }
        
    float L1norm = sum_delta / sum_ref;
    float eps = 1e-6;
    printf("L1 norm: %E\n", L1norm);
    printf((L1norm < eps) ? "TEST PASSED\n" : "TEST FAILED\n");

    /* Compare the optimized results */
    printf("\nComparing optimized CPU and GPU results\n");
    float sum_delta2 = 0, sum_ref2 = 0;
    for (i = 0; i < num_elements; i++) {
        sum_delta2 += fabsf(matrix_a[i] - result_with_optimization[i]);
        sum_ref2   += fabsf(matrix_a[i]);
    }
        
    float L1norm2 = sum_delta2 / sum_ref2;
    printf("L1 norm: %E\n", L1norm2);
    printf((L1norm2 < eps) ? "TEST PASSED\n" : "TEST FAILED\n");

    free(matrix_a);
    free(matrix_c);
    free(result_with_optimization);
    free(gpu_result);
    free(gaussian_kernel);

    exit(EXIT_SUCCESS);
}

/* Check for errors reported by the CUDA run time */
void check_for_error(char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s)\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

    return;
} 

/* Print convolution kernel */
void print_kernel(float *kernel, int half_width)
{
    int i, j = 0;
    for (i = -half_width; i <= half_width; i++) {
        printf("%0.2f ", kernel[j]);
        j++;
    }

    printf("\n");
    return;
}

/* Print matrix */
void print_matrix(float *matrix, int num_cols, int num_rows)
{
    int i,  j;
    float element;
    for (i = 0; i < num_rows; i++) {
        for (j = 0; j < num_cols; j++){
            element = matrix[i * num_cols + j];
            printf("%0.2f ", element);
        }
        printf("\n");
    }

    return;
}

