/* Particle swarm optimizer.
 *
 * Note: This is an implementation of the original algorithm proposed in: 
 *
 * Yuhui Shi, "Particle Swarm Optimization," IEEE Neural Networks Society, pp. 8-13, February, 2004.
 *
 * Compile using provided Makefile: make 
 * If executable exists or if you have made changes to the .h file but not to the .c files, delete the executable and rebuild 
 * as follows: make clean && make
 *
 * Author: Naga Kandasamy
 * Date modified: May 10, 2023 
 *
 * Student/team: Noah Robinson and Jack Pinkstone
 * Date: 5/24/23
 */  
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "pso.h"

int main(int argc, char **argv)
{
    if (argc < 8) {
        fprintf(stderr, "Usage: %s function-name dimension swarm-size xmin xmax max-iter num-threads\n", argv[0]);
        fprintf(stderr, "function-name: name of function to optimize\n");
        fprintf(stderr, "dimension: dimensionality of search space\n");
        fprintf(stderr, "swarm-size: number of particles in swarm\n");
        fprintf(stderr, "xmin, xmax: lower and upper bounds on search domain\n");
        fprintf(stderr, "max-iter: number of iterations to run the optimizer\n");
        fprintf(stderr, "num-threads: number of threads to create\n");
        exit(EXIT_FAILURE);
    }

    char *function = argv[1];
    int dim = atoi(argv[2]);
    int swarm_size = atoi(argv[3]);
    float xmin = atof(argv[4]);
    float xmax = atof(argv[5]);
    int max_iter = atoi(argv[6]);
    int num_threads = atoi(argv[7]);
  
    float seconds;
    struct timeval start, stop;
    /* Optimize using reference version */
    int status;
    gettimeofday(&start, NULL);
    status = optimize_gold(function, dim, swarm_size, xmin, xmax, max_iter);
    gettimeofday(&stop, NULL);
    if (status < 0)
    {
        fprintf(stderr, "Error optimizing function using reference code\n");
        exit (EXIT_FAILURE);
    }
    seconds = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000);
    printf("Gold Time: %f\n", seconds); 

    // OMP Version
    gettimeofday(&start, NULL);
    status = optimize_using_omp(function, dim, swarm_size, xmin, xmax, max_iter, num_threads);
    gettimeofday(&stop, NULL);
    if (status < 0) {
        fprintf(stderr, "Error optimizing function using OpenMP\n");
        exit (EXIT_FAILURE);
    }
    seconds = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000);
    printf("OMP Time: %f\n", seconds);

    exit(EXIT_SUCCESS);
}

/* Print command-line arguments */
void print_args(char *function, int dim, int swarm_size, float xmin, float xmax)
{
    fprintf(stderr, "Function to optimize: %s\n", function);
    fprintf(stderr, "Dimensionality of search space: %d\n", dim);
    fprintf(stderr, "Number of particles: %d\n", swarm_size);
    fprintf(stderr, "xmin: %f\n", xmin);
    fprintf(stderr, "xmax: %f\n", xmax);
    return;
}

