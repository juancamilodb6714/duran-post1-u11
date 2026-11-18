// matMul.cu - Multiplicacion de matrices naive vs tiling en shared memory
// (Unidad 11, Parte 1, Paso 3). El kernel matMulTiled es el de la guia.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE 16

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                             \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Version naive: cada thread lee una fila de A y una columna de B desde memoria global
__global__ void matMulNaive(const float *A, const float *B,
                            float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// Version con tiling: cada bloque carga tiles de TILE x TILE en shared memory
__global__ void matMulTiled(const float *A, const float *B,
                             float *C, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        // Cargar tile de A en shared memory
        if (row < N && t*TILE + threadIdx.x < N)
            sA[threadIdx.y][threadIdx.x] =
                A[row*N + t*TILE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        // Cargar tile de B en shared memory
        if (col < N && t*TILE + threadIdx.y < N)
            sB[threadIdx.y][threadIdx.x] =
                B[(t*TILE + threadIdx.y)*N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();   // Esperar a que el tile este cargado

        for (int k = 0; k < TILE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();   // Esperar antes del siguiente tile
    }

    if (row < N && col < N) C[row*N + col] = sum;
}

// Referencia en CPU, orden i-k-j para que no tarde tanto con N = 1024
void matMulCPU(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N * N; i++) C[i] = 0.0f;
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            float a = A[i * N + k];
            for (int j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
        }
}

// Lanza un kernel REPS veces y devuelve el promedio en ms medido con cudaEvent
static float timeKernel(int tiled, const float *d_A, const float *d_B,
                        float *d_C, int N) {
    const int REPS = 10;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Calentamiento
    if (tiled) matMulTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    else       matMulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < REPS; r++) {
        if (tiled) matMulTiled<<<grid, block>>>(d_A, d_B, d_C, N);
        else       matMulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / REPS;
}

static float maxError(const float *X, const float *Y, int n) {
    float e = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(X[i] - Y[i]);
        if (d > e) e = d;
    }
    return e;
}

static void benchmark(int N) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    float *h_C_naive = (float*)malloc(bytes);
    float *h_C_tiled = (float*)malloc(bytes);

    // Valores pequenos para que la suma en float no pierda mucha precision
    srand(42);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
        h_B[i] = (float)(rand() % 100) / 100.0f;
    }
    matMulCPU(h_A, h_B, h_C_cpu, N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    float naive_ms = timeKernel(0, d_A, d_B, d_C, N);
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, bytes, cudaMemcpyDeviceToHost));
    float tiled_ms = timeKernel(1, d_A, d_B, d_C, N);
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, bytes, cudaMemcpyDeviceToHost));

    double gflop = 2.0 * N * (double)N * N / 1e9;
    printf("N = %4d | naive: %7.3f ms (%7.1f GFLOP/s) | tiled: %7.3f ms "
           "(%7.1f GFLOP/s) | speedup: %.2fx | err naive: %.2e | err tiled: %.2e\n",
           N, naive_ms, gflop / (naive_ms / 1e3), tiled_ms,
           gflop / (tiled_ms / 1e3), naive_ms / tiled_ms,
           maxError(h_C_naive, h_C_cpu, N * N), maxError(h_C_tiled, h_C_cpu, N * N));

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_naive); free(h_C_tiled);
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | TILE = %d | shared por bloque: %zu bytes\n",
           prop.name, TILE, 2 * TILE * TILE * sizeof(float));
    benchmark(512);
    benchmark(1024);
    return 0;
}
