#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <random>
#include <sys/time.h>

void traceMatrixOnHost(double *h_A, double *hostRef, int nx, int ny) {
    if (nx != ny) return;
    hostRef[0] = 0;
    for (int i = 0; i < nx; i++) {
        hostRef[0] += h_A[i*nx + i];
    }
}

// Half as many total threads as data
// Each block sums the diagonal elements separately
// The results from each block is reduced at the host side
__global__ void traceMatrixOnDeviceBasic(double *d_A, double *d_out, int nx) {
    if (blockDim.x * blockIdx.x + threadIdx.x >= nx) return; // Boundary check

    // Start of section of data processed by each block
    double *idata = d_A + (2 * blockIdx.x * blockDim.x) * (nx + 1);
    int index = threadIdx.x * (nx + 1); // Thread index within idata

    for (int stride = (nx + 1) * blockDim.x; stride >= nx + 1; stride >>= 1) {
        if (index < stride) {
            idata[index] += idata[index + stride];
        }
        __syncthreads();
    }

    // Store trace of each block into d_out
    if (threadIdx.x == 0) d_out[blockIdx.x] = idata[0];
}

// Each thread initially handles 8 pieces of data.
__global__ void traceMatrixOnDeviceOptimized(double *d_A, double *d_out, int nx) {
    double *idata = d_A + (8 * blockIdx.x * blockDim.x) * (nx + 1);
    int index = threadIdx.x * (nx + 1); // Thread index within idata

    // Unroll 8
    int jump = blockDim.x * (nx + 1);
    if (index + 7 * jump < 8 * jump) {
        int jump = blockDim.x * (nx + 1);
        double a1 = idata[index];
        double a2 = idata[index + jump];
        double a3 = idata[index + jump * 2];
        double a4 = idata[index + jump * 3];
        double a5 = idata[index + jump * 4];
        double a6 = idata[index + jump * 5];
        double a7 = idata[index + jump * 6];
        double a8 = idata[index + jump * 7];
        idata[index] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    for (int stride = (nx + 1) * blockDim.x / 2; stride >= nx + 1; stride >>= 1) {
        if (index < stride) {
            idata[index] += idata[index + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) d_out[blockIdx.x] = idata[0];
}

// Value of trace is stored in hostRef[0]
int validateData(double *hostRef, double *gpuRef) {
    double epsilon = 1E-3;
    if (fabs(hostRef[0] - gpuRef[0]) > epsilon) return 0;
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
    int nx = (1 << 12);// + (1 << 3);
    int ny = (1 << 12);// + (1 << 3);

    int nxy = nx*ny;
    int nBytes = nxy*sizeof(double);
    printf("Matrix Size: %d x %d\n", nx, ny);

    double *h_A, *hostRef, *gpuRef; // Data is stored in 1D array
    h_A = (double*) malloc(nBytes);
    hostRef = (double*) malloc(nBytes);
    gpuRef = (double*) malloc(nBytes);

    initializeData(h_A, nxy, 1);

    // Compute trace on CPU for result checking
    traceMatrixOnHost(h_A, hostRef, nx, ny);
    
    // Malloc Data on Device
    double *d_A, *d_out;
    cudaMalloc((double**)&d_A, nBytes);
    cudaMalloc((double**)&d_out, nBytes);

    // Transfer input data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    int dimx = 128; // Block dimension
    dim3 block(dimx);
    dim3 grid((nx + block.x * 8 - 1) / (block.x * 8)); // 1D grid, 1D blocks.

    // Invoke and time kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    traceMatrixOnDeviceOptimized <<< grid, block >>> (d_A, d_out, nx);
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
    int valid = validateData(hostRef, gpuRef);
    if (!valid) {
        printf("GPU result (%f) does not match CPU result (%f).\n", gpuRef[0], hostRef[0]);
    } else {
        printf("GPU result matches.\n");
    }
    
    return 0;
}