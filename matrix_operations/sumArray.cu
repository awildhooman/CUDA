#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <random>
#include <sys/time.h>

#ifndef DATA_PER_THREAD
#define DATA_PER_THREAD 16
#endif

#define ARG_NX 1
#define ARG_NY 2
#define ARG_SEED1 3
#define ARG_SEED2 4
#define ARG_THREADS 5

#define CHECK(call)                                                        \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

void sumMatrixOnHost(float *h_A, float *h_B, float *h_C, int nxy) {
    for (int i = 0; i < nxy; i++) {
        h_C[i] = h_A[i] + h_B[i];
    }
}

// Each thread handles just 1 set of data
__global__ void sumMatrixOnDeviceBasic(float *d_A, float *d_B, float *d_C, int nx, int ny) {
    int index = nx * blockDim.y * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nx*ny) d_C[index] = d_A[index] + d_B[index];
}

// Each thread handles DATA_PER_THREAD amount of data
__global__ void sumMatrixOnDeviceOptimized(float *d_A, float *d_B, float *d_C, int nxy) {
    int base = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll
    for (int i = 0; i < DATA_PER_THREAD; i++) {
        int index = base + blockDim.x * i;
        if (index < nxy) d_C[index] = d_A[index] + d_B[index];
    }
}

int validateData(float *hostRef, float *gpuRef, int nxy) {
    float epsilon = 1E-4;
    for (int i = 0; i < nxy; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) return 0;
    }
    return 1;
}

// Initialize data with random floats between 0 and 1000. Input seed for consistent values, 0 for time-based rng
void initializeData(float *data, int size, unsigned int seed) {
    time_t t;
    if (seed == 0) {
        srand((unsigned) time(&t));
    } else {
        srand(seed);
    }
    for (int i=0; i<size; i++) {
        data[i] = ((float)rand() / (float)RAND_MAX) * 1000.0;
    }
}

int main(int argc, char **argv) {
    // Function args: nx, ny, seed 1, seed 2, threads/block
    if (argc <= 1) {
        printf("Error: command-line arguments required: nx, ny, seed1, seed2, threads/block");
        return 1;
    }
    printf("%s Starting...\n", argv[0]);

    // Set up Device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // Malloc & Initialize Data on Host
    int nx = atoi(argv[ARG_NX]);
    int ny = atoi(argv[ARG_NY]);

    int nxy = nx*ny;
    int nBytes = nxy*sizeof(float);
    printf("Matrix Size: %d x %d\n", nx, ny);

    float *h_A, *h_B, *hostRef, *gpuRef; // Data is stored in 1D array
    h_A = (float*) malloc(nBytes);
    h_B = (float*) malloc(nBytes);
    hostRef = (float*) malloc(nBytes);
    gpuRef = (float*) malloc(nBytes);

    initializeData(h_A, nxy, atoi(argv[ARG_SEED1]));
    initializeData(h_B, nxy, atoi(argv[ARG_SEED2]));

    // Add matrices on CPU for result checking
    sumMatrixOnHost(h_A, h_B, hostRef, nxy);
    
    // Malloc Data on Device
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    // Transfer input data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    int dimx = atoi(argv[ARG_THREADS]); // Block dimension
    dim3 block(dimx);
    dim3 grid( (nxy + block.x * DATA_PER_THREAD - 1) / (block.x * DATA_PER_THREAD) ); // 1D grid, 1D blocks

    // Invoke and time kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sumMatrixOnDeviceOptimized <<< grid, block >>> (d_A, d_B, d_C, nxy);
    cudaEventRecord(stop);
    cudaDeviceSynchronize(); // Wait until stop event is recorded
    float timeElapsedMs = 0;
    cudaEventElapsedTime(&timeElapsedMs, start, stop);
    printf("Device elapsed time: %g ms\n", timeElapsedMs);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // Validate
    int valid = validateData(hostRef, gpuRef, nxy);
    if (!valid) {
        printf("GPU result does not match CPU result.\n");
    } else {
        printf("GPU result matches.\n");
    }

    // Free memory and reset
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();

    return 0;
}