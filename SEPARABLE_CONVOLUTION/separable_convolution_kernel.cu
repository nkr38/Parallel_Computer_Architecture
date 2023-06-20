/* FIXME: Edit this file to complete the functionality of 2D separable 
 * convolution on the GPU. You may add additional kernel functions 
 * as necessary. 
 */

__global__ void convolve_rows_kernel_naive(float *result, float *input, float *kernel, int num_cols, int num_rows, int half_width)
{
    int i, i1, j, j1, j2, x, y;

    x = blockDim.x * blockIdx.x + threadIdx.x;
    y = blockDim.y * blockIdx.y + threadIdx.y;

    j1 = x - half_width;
    j2 = x + half_width;

    /* Clamp at the edges of the matrix */
    j1 = max(0, j1);
    j2 = min(num_rows - 1, j2);

    /* Obtain relative position of starting element from element being convolved */
    i1 = j1 - x;

    j1 = j1 - x + half_width; /* Obtain operating width of the kernel */
    j2 = j2 - x + half_width;

    /* Convolve along row */
    result[y * num_cols + x] = 0.0f;
    for(i = i1, j = j1; j <= j2; j++, i++)
        result[y * num_cols + x] += kernel[j] * input[y * num_cols + x + i];

    return;
}

__global__ void convolve_columns_kernel_naive(float *result, float *input, float *kernel, int num_cols, int num_rows, int half_width)
{
    int i, i1, j, j1, j2, x, y;

    x = blockDim.x * blockIdx.x + threadIdx.x;
    y = blockDim.y * blockIdx.y + threadIdx.y;

    j1 = y - half_width;
    j2 = y + half_width;

    /* Clamp at the edges of the matrix */
    j1 = max(0, j1);
    j2 = min(num_rows - 1, j2);

    /* Obtain relative position of starting element from element being convolved */
    i1 = j1 - y;

    j1 = j1 - y + half_width; /* Obtain the operating width of the kernel.*/
    j2 = j2 - y + half_width;

    /* Convolve along column */
    result[y * num_cols + x] = 0.0f;
    for (i = i1, j = j1; j <= j2; j++, i++)
        result[y * num_cols + x] += kernel[j] * input[y * num_cols + x + (i * num_cols)];

    return;
}

__global__ void convolve_rows_kernel_optimized(float *result, float *input, int num_cols, int num_rows, int half_width)
{
    __shared__ float inputShared[(THREAD_BLOCK_SIZE + HALF_WIDTH * 2) * THREAD_BLOCK_SIZE];
    int i, i1, j, j1, j2, x, y;

    x = blockDim.x * blockIdx.x + threadIdx.x;
    y = blockDim.y * blockIdx.y + threadIdx.y;

    int leftHaloIndex = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    int rightHaloIndex = (blockIdx.x + 1) * blockDim.x + threadIdx.x;

    /* Load input elements into shared memory with halo regions */
    if (threadIdx.x >= (blockDim.x - half_width)) 
    {
        inputShared[threadIdx.y * blockDim.y + (threadIdx.x - (blockDim.x - half_width))] = (leftHaloIndex < 0) ? 0.0 : input[leftHaloIndex + y * num_rows];
    }

    if (x < num_cols)
        inputShared[threadIdx.y * blockDim.y + (threadIdx.x + half_width)] = input[y * num_rows + x];
    else
        inputShared[threadIdx.y * blockDim.y + (threadIdx.x + half_width)] = 0.0;

    if (threadIdx.x < half_width) 
    {
        inputShared[threadIdx.y * blockDim.y + threadIdx.x + (blockDim.x + half_width)] = (rightHaloIndex >= num_cols) ? 0.0 : input[rightHaloIndex + y * num_rows];
    }

    __syncthreads();

    j1 = x - half_width;
    j2 = x + half_width;

    j1 = max(0, j1);
    j2 = min(num_rows - 1, j2);

    /* Obtain relative position of starting element from element being convolved */
    i1 = j1 - x;

    j1 = j1 - x + half_width; /* Obtain operating width of the kernel */
    j2 = j2 - x + half_width;

    /* Convolve along row */
    result[y * num_cols + x] = 0.0f;
    for (i = i1, j = j1; j <= j2; j++, i++)
        result[y * num_cols + x] += kernel_optimized[j] * inputShared[threadIdx.y * blockDim.x + (threadIdx.x + half_width) + i];
    
    return;
}

__global__ void convolve_columns_kernel_optimized(float *result, float *input, int num_cols, int num_rows, int half_width)
{
    __shared__ float inputShared[(THREAD_BLOCK_SIZE + HALF_WIDTH * 2) * THREAD_BLOCK_SIZE];
    int i, i1, j, j1, j2, x, y;

    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;

    int leftHaloIndex = (blockIdx.y - 1) * blockDim.y + threadIdx.y;
    int rightHaloIndex = (blockIdx.y + 1) * blockDim.y + threadIdx.y;

    /* Load input elements into shared memory with halo regions */
    if (threadIdx.y >= (blockDim.y - half_width)) 
    {
        inputShared[(threadIdx.y - (blockDim.y - half_width)) * blockDim.y + threadIdx.x] = (leftHaloIndex < 0) ? 0.0 : input[leftHaloIndex * num_cols + x];
    }

    if (y < num_rows)
        inputShared[(threadIdx.y + half_width) * blockDim.y + threadIdx.x] = input[y * num_cols + x];
    else
        inputShared[(threadIdx.y + half_width) * blockDim.y + threadIdx.x] = 0.0;

    if (threadIdx.y < half_width) 
    {
        inputShared[(threadIdx.y + (blockDim.y + half_width)) * blockDim.y + threadIdx.x] = (rightHaloIndex >= num_rows) ? 0.0 : input[rightHaloIndex * num_cols + x];
    }
 
    __syncthreads();

    j1 = y - half_width;
    j2 = y + half_width;

    j1 = max(0, j1);
    j2 = min(num_rows - 1, j2);

    /* Obtain relative position of starting element from element being convolved */
    i1 = j1 - y; 
    
    j1 = j1 - y + half_width; /* Obtain the operating width of the kernel.*/
    j2 = j2 - y + half_width;

    /* Convolve along column */
    result[y * num_cols + x] = 0.0f;
    for (i = i1, j = j1; j <= j2; j++, i++)
        result[y * num_cols + x] += kernel_optimized[j] * inputShared[(threadIdx.y + half_width) * blockDim.x + threadIdx.x + (i * blockDim.x)];
    
    return;
}



