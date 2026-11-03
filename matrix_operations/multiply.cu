#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

#define ARG_M 1
#define ARG_N 2
#define ARG_P 3
#define ARG_SEED1 4
#define ARG_SEED2 5

#define CHECK(call)                                                        \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// C (M x P) = A (M x N) * B (N x P)
void multiplyMatrixOnHost(const double *A, const double *B, double *C, int M, int N, int P) {
    for (int r = 0; r < M; ++r) {
        for (int p = 0; p < P; ++p) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[r * N + k] * B[k * P + p];
            }
            C[r * P + p] = sum;
        }
    }
}

// Each thread corresponds to one piece of data in the matrix.
// Compute tile by tile to make use of memory caching
__global__ void matrixMulRectangular(const double *A, const double *B, double *C, int M, int N, int P) {
    __shared__ double s_A[TILE_SIZE][TILE_SIZE];
    __shared__ double s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // Row in A and C
    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // Column in B and C

    double value = 0.0;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < N) {
            s_A[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0;
        }

        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < N && col < P) {
            s_B[threadIdx.y][threadIdx.x] = B[b_row * P + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();
    
        // Dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            value += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < P) {
        C[row * P + col] = value;
    }
}

// Compares two rows x cols matrices. Uses a relative tolerance because
// the sums are large (~1e9) and CPU/GPU accumulate in different orders.
int validateData(const double *hostRef, const double *gpuRef, int rows, int cols) {
    const double relEps = 1e-9;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double h = hostRef[i * cols + j];
            double g = gpuRef[i * cols + j];
            double scale = fmax(fabs(h), 1.0);
            if (fabs(h - g) > relEps * scale) {
                printf("Mismatch at (%d, %d): host = %g, gpu = %g\n", i, j, h, g);
                return 0;
            }
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
    for (int i = 0; i < size; i++) {
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

    // Input args: M, N, P, seed 1, seed 2
    if (argc < 6) {
        printf("Error: command-line arguments required: M, N, P, seed1, seed2");
        return 1;
    }

    // Set up Device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // Matrix dimensions: A is M x N, B is N x P, C is M x P.
    int M = atoi(argv[ARG_M]);
    int N = atoi(argv[ARG_N]);
    int P = atoi(argv[ARG_P]);
    printf("Matrix Sizes: A = %d x %d, B = %d x %d, C = %d x %d\n",
           M, N, N, P, M, P);

    int sizeA = M * N;
    int sizeB = N * P;
    int sizeC = M * P;
    int bytesA = sizeA * sizeof(double);
    int bytesB = sizeB * sizeof(double);
    int bytesC = sizeC * sizeof(double);

    // Malloc & initialize data on host
    double *h_A    = (double*) malloc(bytesA);
    double *h_B    = (double*) malloc(bytesB);
    double *hostRef = (double*) malloc(bytesC);
    double *gpuRef  = (double*) malloc(bytesC);
    if (!h_A || !h_B || !hostRef || !gpuRef) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }

    initializeData(h_A, sizeA, atoi(argv[ARG_SEED1]));
    initializeData(h_B, sizeB, atoi(argv[ARG_SEED2]));

    // Compute reference result on CPU
    struct timeval cpuStart, cpuEnd;
    gettimeofday(&cpuStart, NULL);
    multiplyMatrixOnHost(h_A, h_B, hostRef, M, N, P);
    gettimeofday(&cpuEnd, NULL);
    double cpuMs = (cpuEnd.tv_sec - cpuStart.tv_sec) * 1000.0 +
                   (cpuEnd.tv_usec - cpuStart.tv_usec) / 1000.0;
    printf("Host elapsed time: %g ms\n", cpuMs);

    // Malloc data on device
    double *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void**)&d_A, bytesA));
    CHECK(cudaMalloc((void**)&d_B, bytesB));
    CHECK(cudaMalloc((void**)&d_C, bytesC));

    // Transfer input data from host to device
    CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    // 2D grid and 2D blocks
    // Each block is one tile
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((P + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);
    printf("Grid: (%d, %d), Block: (%d, %d)\n", grid.x, grid.y, block.x, block.y);

    // Invoke and time kernel
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    matrixMulRectangular<<<grid, block>>>(d_A, d_B, d_C, M, N, P);
    CHECK(cudaGetLastError());          // catch launch errors
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));  // wait until stop event is recorded
    float timeElapsedMs = 0;
    CHECK(cudaEventElapsedTime(&timeElapsedMs, start, stop));
    printf("Device elapsed time: %g ms\n", timeElapsedMs);

    // Copy result back to host
    CHECK(cudaMemcpy(gpuRef, d_C, bytesC, cudaMemcpyDeviceToHost));

    // Validate
    int valid = validateData(hostRef, gpuRef, M, P);
    if (!valid) {
        printf("GPU result does not match CPU result.\n");
    } else {
        printf("GPU result matches.\n");
    }

    // Cleanup
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    CHECK(cudaDeviceReset());

    return valid ? 0 : 1;
}
