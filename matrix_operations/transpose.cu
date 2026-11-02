#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <random>
#include <sys/time.h>

#define DATA_PER_THREAD 8

void transposeMatrixOnHost(double *h_A, double *hostRef, int nx, int ny) {
    for (int r = 0; r < ny; ++r) {
        for (int c = 0; c < nx; ++c) {
            int orig_idx  = r * nx + c; // Row-major index in original
            int trans_idx = c * ny + r; // Row-major index in transposed matrix

            hostRef[trans_idx] = h_A[orig_idx];
        }
    }
}

// 1 thread per data
__global__ void transposeMatrixOnDeviceBasic(double *d_A, double *d_out, int nx, int ny) {
    int r = blockIdx.y * blockDim.y + threadIdx.y; // Row in original
    int c = blockIdx.x * blockDim.x + threadIdx.x; // Column in original

    if (r < ny && c < nx) {
        int orig_idx = r * nx + c;
        int trans_idx = c * ny + r;
        
        d_out[trans_idx] = d_A[orig_idx];
    }
    
}

// Each thread handles multiple data
__global__ void transposeMatrixOnDeviceOptimized(double *d_A, double *d_out, int nx, int ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y_base = blockIdx.y * DATA_PER_THREAD;

    if (x < nx) {
        #pragma unroll
        for (int i = 0; i < DATA_PER_THREAD; i++) {
            int y = y_base + i;
            if (y < ny) {
                int orig_idx = y * nx + x;
                int trans_idx = x * ny + y;

                d_out[trans_idx] = d_A[orig_idx];
            }
        }
    }
}

// nx and ny are associated with the transpose
int validateData(double *hostRef, double *gpuRef, int nx, int ny) {
    double epsilon = 1E-4;
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            if (fabs(hostRef[i * ny + nx] - gpuRef[i * ny + nx]) > epsilon) return 0;
        }
    }
    return 1;
}

// Initialize data with random doubles between 0 and 1000. Input seed for consistent values, 0 for time-based rng
void initializeData(double *data, int size, unsigned int seed) {
    time_t t;
    if (seed == 0) {
        srand((unsigned) time(&t));
    } else {
        srand(seed);
    }
    for (int i=0; i<size; i++) {
        data[i] = ((double)rand() / (double)RAND_MAX) * 1000.0;
    }
}

void printMatrix(double *data, int nx, int ny) {
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            printf("%g ", data[i * nx + j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);

    // Set up Device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // Malloc & Initialize Data on Host
    int nx = (1 << 10);
    int ny = (1 << 10) + (1 << 2);

    int nxy = nx*ny;
    int nBytes = nxy*sizeof(double);
    printf("Matrix Size: %d x %d\n", nx, ny);

    double *h_A, *hostRef, *gpuRef; // Data is stored in 1D array
    h_A = (double*) malloc(nBytes);
    hostRef = (double*) malloc(nBytes);
    gpuRef = (double*) malloc(nBytes);

    initializeData(h_A, nxy, 1);

    // Compute trace on CPU for result checking
    transposeMatrixOnHost(h_A, hostRef, nx, ny);
    
    // Malloc Data on Device
    double *d_A, *d_out;
    cudaMalloc((double**)&d_A, nBytes);
    cudaMalloc((double**)&d_out, nBytes);

    // Transfer input data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    // 2D grid, 1D blocks.
    int dimx = 128;
    dim3 block(dimx);
    dim3 grid((nx + block.x - 1) / block.x, (ny+DATA_PER_THREAD-1) / DATA_PER_THREAD);

    // Invoke and time kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    transposeMatrixOnDeviceOptimized <<< grid, block >>> (d_A, d_out, nx, ny);
    cudaEventRecord(stop);
    cudaDeviceSynchronize(); // Wait until stop event is recorded
    float timeElapsedMs = 0;
    cudaEventElapsedTime(&timeElapsedMs, start, stop);
    printf("Device elapsed time: %g ms\n", timeElapsedMs);
    cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost);

    // Reduce at host side sequentially
    for (int i = 1; i < grid.x; i++) {
        gpuRef[0] += gpuRef[i];
    }
    // Validate
    int valid = validateData(hostRef, gpuRef, ny, nx);
    if (!valid) {
        printf("GPU result does not match CPU result.\n");
    } else {
        printf("GPU result matches.\n");
    }
    
    return 0;
}