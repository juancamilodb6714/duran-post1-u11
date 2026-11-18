// vectorAdd.cu - Suma de vectores CPU vs GPU (Unidad 11, Parte 1, Paso 2)
// Kernel y benchmark de la guia. Se agrego un bucle para medir N = 1M, 4M y 16M,
// el tiempo GPU total con cudaMemcpy y el chequeo de errores con cudaGetLastError.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// Revisa cada llamada a la API de CUDA y corta el programa si algo falla
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                             \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel CUDA: cada thread calcula un elemento de C
__global__ void vectorAdd(const float *A, const float *B,
                          float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

// Suma de vectores en CPU (referencia)
void vectorAddCPU(const float *A, const float *B,
                  float *C, int n) {
    for (int i = 0; i < n; i++) C[i] = A[i] + B[i];
}

// Corre el benchmark completo para un tamano n
static void benchmark(int n) {
    size_t bytes = (size_t)n * sizeof(float);

    // Asignar memoria en host
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes);

    // Inicializar vectores
    for (int i = 0; i < n; i++) {
        h_A[i] = (float)i * 0.5f;
        h_B[i] = (float)i * 1.5f;
    }

    // --- Benchmark CPU ---
    clock_t t0 = clock();
    vectorAddCPU(h_A, h_B, h_C_cpu, n);
    double cpu_ms = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0;

    // --- Benchmark GPU ---
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    cudaEvent_t start, stop, totStart, totStop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&totStart));
    CUDA_CHECK(cudaEventCreate(&totStop));

    int blockSize = 256;
    int gridSize  = (n + blockSize - 1) / blockSize;

    // Calentamiento para que la primera medicion no cargue el arranque del contexto
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Tiempo total: H2D + kernel + D2H
    CUDA_CHECK(cudaEventRecord(totStart));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Tiempo solo del kernel
    CUDA_CHECK(cudaEventRecord(start));
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(totStop));
    CUDA_CHECK(cudaEventSynchronize(totStop));

    float gpu_ms = 0, total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, totStart, totStop));

    // Verificar resultados
    int errors = 0;
    for (int i = 0; i < n; i++)
        if (fabs(h_C_gpu[i] - h_C_cpu[i]) > 1e-4f) errors++;

    printf("N = %8d (%2dM) | CPU: %8.2f ms | GPU kernel: %6.2f ms | "
           "GPU total: %7.2f ms | Errores: %d\n",
           n, n >> 20, cpu_ms, gpu_ms, total_ms, errors);

    // Liberar memoria
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(totStart));
    CUDA_CHECK(cudaEventDestroy(totStop));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (cc %d.%d, %d SM)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);

    int sizes[] = {1 << 20, 1 << 22, 1 << 24};   // 1M, 4M y 16M elementos
    for (int s = 0; s < 3; s++) benchmark(sizes[s]);
    return 0;
}
