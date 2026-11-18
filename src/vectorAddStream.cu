// vectorAddStream.cu - Suma de vectores con streams y cudaMemcpyAsync
// (Unidad 11, Parte 1, Paso 4). Codigo de la guia mas verificacion contra CPU,
// chequeo de errores y NSTREAMS cambiable al compilar con -DNSTREAMS=8.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N (1 << 24)        // 16M elementos, igual que vectorAdd.cu
#ifndef NSTREAMS
#define NSTREAMS 4
#endif
#define CHUNK (N / NSTREAMS)

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                             \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

__global__ void vectorAddChunk(const float *A, const float *B,
                               float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

// Compara h_C contra la suma hecha en CPU, igual que en vectorAdd.cu
static int countErrors(const float *h_A, const float *h_B, const float *h_C) {
    int errors = 0;
    for (int i = 0; i < N; i++)
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-4f) errors++;
    return errors;
}

int main() {
    size_t bytes      = N * sizeof(float);
    size_t chunkBytes = CHUNK * sizeof(float);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | copy engines (asyncEngineCount): %d | NSTREAMS = %d\n",
           prop.name, prop.asyncEngineCount, NSTREAMS);

    // Memoria host "pinned" (page-locked): obligatoria para que
    // cudaMemcpyAsync se ejecute realmente de forma asincrona
    float *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaHostAlloc(&h_A, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_B, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_C, bytes, cudaHostAllocDefault));

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i * 0.5f;
        h_B[i] = (float)i * 1.5f;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    int blockSize = 256;
    int chunkGrid = (CHUNK + blockSize - 1) / blockSize;

    // Calentamiento del contexto para no medir el arranque
    vectorAddChunk<<<(N + blockSize - 1) / blockSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Baseline sincrono: una sola transferencia H2D, un kernel, una D2H ---
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    vectorAddChunk<<<(N + blockSize - 1) / blockSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float syncMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&syncMs, start, stop));
    printf("Sincrono (H2D + kernel + D2H completos): %.2f ms | Errores: %d\n",
           syncMs, countErrors(h_A, h_B, h_C));

    // Se limpia h_C para que la verificacion de streams no use el resultado anterior
    for (int i = 0; i < N; i++) h_C[i] = 0.0f;
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    // --- Version con streams: solapa transferencia y computo por chunks ---
    cudaStream_t streams[NSTREAMS];
    for (int s = 0; s < NSTREAMS; s++) CUDA_CHECK(cudaStreamCreate(&streams[s]));

    CUDA_CHECK(cudaEventRecord(start));
    for (int s = 0; s < NSTREAMS; s++) {
        size_t off = (size_t)s * CHUNK;
        CUDA_CHECK(cudaMemcpyAsync(d_A + off, h_A + off, chunkBytes,
                                   cudaMemcpyHostToDevice, streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(d_B + off, h_B + off, chunkBytes,
                                   cudaMemcpyHostToDevice, streams[s]));
        vectorAddChunk<<<chunkGrid, blockSize, 0, streams[s]>>>(
            d_A + off, d_B + off, d_C + off, CHUNK);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h_C + off, d_C + off, chunkBytes,
                                   cudaMemcpyDeviceToHost, streams[s]));
    }
    for (int s = 0; s < NSTREAMS; s++) CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float asyncMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&asyncMs, start, stop));
    printf("Con streams (%d, solapado): %.2f ms | Errores: %d\n",
           NSTREAMS, asyncMs, countErrors(h_A, h_B, h_C));
    printf("Reduccion: %.1f%%\n", 100.0 * (syncMs - asyncMs) / syncMs);

    for (int s = 0; s < NSTREAMS; s++) CUDA_CHECK(cudaStreamDestroy(streams[s]));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A)); CUDA_CHECK(cudaFreeHost(h_B)); CUDA_CHECK(cudaFreeHost(h_C));
    return 0;
}
