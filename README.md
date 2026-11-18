# Post-Contenido 1 — CUDA Benchmark CPU vs GPU

**Arquitectura de Computadores · Unidad 11 · Post-contenido**

| Dato | Valor |
|---|---|
| Estudiante | Juan Camilo Durán Bohórquez |
| Código | 1152532 |
| Repositorio | duran-post1-u11 |

## Descripción del laboratorio

El laboratorio tiene dos partes en el mismo repositorio. En la Parte 1 se escribieron tres programas CUDA en C, una suma de vectores, una multiplicación de matrices con tiling en shared memory y una suma de vectores con streams y `cudaMemcpyAsync`, y se midió todo contra la CPU con `cudaEvent`. En la Parte 2 se hizo un análisis comparativo de GPU, TPU, NPU y FPGA con datos reales de MLPerf Inference v4.1, que está en [docs/analisis-comparativo.md](docs/analisis-comparativo.md).

## Estructura del repositorio

```
duran-post1-u11/
├── README.md
├── src/
│   ├── vectorAdd.cu          Paso 2, suma de vectores CPU vs GPU
│   ├── matMul.cu             Paso 3, naive vs tiling en shared memory
│   └── vectorAddStream.cu    Paso 4, síncrono vs streams
├── docs/
│   └── analisis-comparativo.md   Parte 2
├── resultados/               salida de texto de cada corrida
└── capturas/                 capturas de las celdas de Colab
```

## Entorno

| Elemento | Valor |
|---|---|
| Plataforma | Google Colab con entorno de GPU T4 |
| GPU | NVIDIA Tesla T4, compute capability 7.5, 40 SM, 15360 MiB GDDR6, PCIe Gen3 x16 |
| Driver | 580.82.07 |
| CUDA | nvcc 13.0, V13.0.88 |
| CPU del host | Intel Xeon a 2.00 GHz con 2 vCPU y 12 GB de RAM |
| Sistema | Ubuntu 24.04.4 LTS |
| Compilador host | gcc 13.3.0 |

El equipo del estudiante tiene solo una Intel Iris Xe y no tiene WSL, entonces la Parte 1 se corrió en Google Colab como dice la guía. Colab ya trae CUDA 13.0 y no la 12.x de los prerrequisitos, pero el código usa solo la API de runtime clásica y compiló igual sin warnings. Para que lo que corrió en Colab sea lo mismo que está en el repo se sacó el `md5sum` de los tres `.cu` en Colab y coincidió con el de los archivos locales, se ve en la captura `p1_cp0_md5_fuentes.png`.

![Entorno de Colab](capturas/p1_cp0_entorno.png)
![md5 de los fuentes](capturas/p1_cp0_md5_fuentes.png)

## Compilación y ejecución

```bash
nvcc -O2 -o vectorAdd src/vectorAdd.cu && ./vectorAdd
nvcc -O2 -o matMul src/matMul.cu && ./matMul
nvcc -O2 -o vectorAddStream src/vectorAddStream.cu && ./vectorAddStream
# barrido del Paso 6, NSTREAMS se cambia al compilar
for s in 1 2 4 8 16 32; do nvcc -O2 -DNSTREAMS=$s -o vas_$s src/vectorAddStream.cu && ./vas_$s; done
```

Los kernels son los de la guía. Lo que se agregó es la macro `CUDA_CHECK`, que revisa cada llamada a la API y `cudaGetLastError()` después de cada lanzamiento, un lanzamiento de calentamiento antes de medir para no contar el arranque del contexto, la verificación contra CPU en `vectorAddStream.cu` y el `#ifndef NSTREAMS` para poder hacer el barrido. En `vectorAdd.cu` se agregaron `#include <math.h>` y `#include <time.h>`, que el código de la guía usa con `fabs` y `clock` pero no incluye.

---

# Parte 1: Benchmark CUDA CPU vs GPU

## Paso 2: Suma de vectores

Se corrió para 1M, 4M y 16M elementos. El tiempo de kernel se mide con `cudaEvent` solo alrededor del lanzamiento, y el tiempo GPU total va desde antes de las dos copias H2D hasta después de la copia D2H.

| N | CPU | GPU kernel | GPU total con `cudaMemcpy` | Kernel contra CPU | Total contra CPU | Errores |
|---|---|---|---|---|---|---|
| 1M | 2.51 ms | 0.05 ms | 5.66 ms | 50x | 0.44x | 0 |
| 4M | 11.70 ms | 0.20 ms | 19.07 ms | 58x | 0.61x | 0 |
| 16M | 42.14 ms | 0.76 ms | 76.20 ms | 55x | 0.55x | 0 |

Lo primero que llama la atención es que el kernel salió más rápido que lo que dice la guía, 0.76 ms contra unos 1.8 ms. Haciendo la cuenta, el kernel con 16M lee 8 bytes y escribe 4 por elemento, son 201 MB en 0.76 ms, unos 265 GB/s, y la T4 tiene 320 GB/s de memoria, o sea que el kernel va al 83% del ancho de banda de la tarjeta. Lo otro que salió raro es el tiempo total, 76 ms, mucho más que lo que tarda la CPU sola. Eso pasa porque `vectorAdd.cu` usa `malloc` normal, memoria paginable, y el driver tiene que copiar primero a un buffer pinned y de ahí a la GPU, la transferencia efectiva sale a unos 2.6 GB/s. En el Paso 4 con `cudaHostAlloc` los mismos 201 MB pasaron en 16.8 ms, más de 4 veces más rápido.

**Checkpoint 1**

![vectorAdd CPU vs GPU](capturas/p1_cp1_vectorAdd.png)
