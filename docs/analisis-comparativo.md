# Análisis comparativo de aceleradores de IA: GPU, TPU, NPU y FPGA

**Arquitectura de Computadores · Unidad 11 · Post-contenido 1, Parte 2**

Juan Camilo Durán Bohórquez, código 1152532

## Introducción

Este documento compara cuatro familias de aceleradores para inteligencia artificial, la GPU de NVIDIA, la TPU de Google, la NPU que viene dentro de los chips de Apple y Qualcomm y la FPGA de AMD Xilinx. La idea es ver en qué se diferencian por dentro, cuánto rinden en MLPerf Inference v4.1, cuánto cuestan y para qué tipo de organización sirve cada una. Al principio parecía que bastaba con mirar los TFLOPS de cada hoja de datos, pero revisando los resultados se vio que los números de marketing salen con sparsity o con INT8 y que la comparación justa es la de MLPerf, donde todos corren el mismo modelo con la misma exactitud mínima.

Los datos de rendimiento no son estimados. Se descargó el repositorio oficial de resultados de MLPerf Inference v4.1 de MLCommons y se leyó el archivo `mlperf_log_summary.txt` de cada sistema, todos con `Result is : VALID` en la división Closed. Para la potencia se promediaron las lecturas del medidor Yokogawa que vienen en los archivos `spl.txt` de los envíos de potencia. Las especificaciones salen de las páginas oficiales de cada fabricante y los precios de las páginas de venta o de la lista de precios de Google Cloud consultadas el 18 de noviembre de 2026.

## 1. Caracterización de cada acelerador

| Acelerador | Paradigma | Unidad de cómputo | Memoria | Pico de cómputo | Potencia | Programación |
|---|---|---|---|---|---|---|
| GPU NVIDIA H100 SXM | SIMT masivo | 132 SM con CUDA Cores y Tensor Cores de 4.ª generación | 80 GB HBM3 a 3.35 TB/s | 989 TFLOPS BF16 denso, 1979 TOPS INT8 denso | hasta 700 W | CUDA, cuDNN, TensorRT |
| TPU v6e Trillium | Systolic array | 1 TensorCore con 2 MXU de 256×256 | 32 GB HBM a 1638 GB/s | 918 TFLOPS BF16, 1836 TOPS INT8 | no publicada por chip | XLA, JAX, TensorFlow, PyTorch XLA |
| TPU v5e | Systolic array | 1 TensorCore con 4 MXU de 128×128 | 16 GB HBM a 819 GB/s | 197 TFLOPS BF16, 393 TOPS INT8 | no publicada por chip | XLA, JAX |
| NPU Apple M3 Ultra | Dataflow fijo | Neural Engine de 32 núcleos | memoria unificada de 96 a 512 GB a más de 800 GB/s | no publicado en TOPS | SoC completo | Core ML, BNNS |
| FPGA AMD Alveo U55C | Reconfigurable | 1304K LUT, 2607K registros, 9024 DSP | 16 GB HBM2 a 460 GB/s, SRAM interna a 35 TB/s | depende del diseño cargado | 115 W típica, 150 W máxima | Vitis HLS, VHDL, Verilog |

Fuentes de la tabla, NVIDIA (n.d.), Google Cloud (n.d.-a, n.d.-b), Apple (2025) y AMD (n.d.). Las cifras de la H100 que publica NVIDIA vienen con sparsity, 1979 TFLOPS BF16 y 3958 TOPS INT8, entonces aquí se dejaron a la mitad para tener el valor denso, que es el que usa un modelo normal.

**GPU.** La GPU ejecuta miles de hilos agrupados en warps de 32 que hacen la misma instrucción sobre datos distintos. Los Tensor Cores hacen multiplicaciones de matrices pequeñas en un ciclo y son los que dan casi todo el rendimiento de IA. La jerarquía de memoria va de registros a shared memory y L1 por SM, luego L2 compartida y al final HBM3. Es la más flexible porque cualquier kernel escrito en CUDA corre ahí, por eso domina el entrenamiento.

**TPU.** La TPU no tiene miles de hilos sino un arreglo sistólico, una malla de multiplicadores donde los datos pasan de celda en celda y cada celda acumula su producto, así se reusa cada peso muchas veces sin volver a la memoria. En la v6e cada MXU es de 256×256, cuatro veces más grande que la de 128×128 de la v5e, y eso explica buena parte del salto de 197 a 918 TFLOPS. Los chips se unen con ICI a 800 GB/s y en la v4 Google ya armaba pods de 4096 chips con conmutadores ópticos (Jouppi et al., 2023). La limitación es que solo se programa a través de XLA y solo se alquila en Google Cloud.

**NPU.** La NPU es un bloque fijo dentro de un SoC de celular o laptop, pensado para inferencia con muy poca energía. Comparte la memoria unificada con la CPU y la GPU, entonces no hay copia por PCIe. Apple no publica TOPS para el M3 Ultra, solo dice que tiene 32 núcleos de Neural Engine, el doble que un M3 Max. Se programa con Core ML, que decide qué capas van a la NPU y cuáles caen en la GPU o la CPU.

**FPGA.** La FPGA no trae una unidad de cómputo fija, el diseñador arma el circuito con LUT, DSP y BRAM. Para inferencia se arma un pipeline donde cada capa del modelo es un bloque de hardware y los datos fluyen sin sistema operativo de por medio, por eso la latencia es muy baja y siempre igual. El costo es que diseñar ese circuito con HLS o VHDL toma semanas y cambiar el modelo implica volver a sintetizar.

## 2. Benchmarks MLPerf Inference v4.1

Se tomaron los tres modelos que pide la guía. En el escenario Offline el generador de carga manda todas las muestras de una vez y se mide el throughput puro. En el escenario Server las consultas llegan con distribución de Poisson y el sistema tiene que cumplir un límite de latencia p99, para BERT ese límite es de 130 ms y para ResNet-50 de 15 ms.

### ResNet-50 v1.5

| Sistema | Envío | Offline muestras/s | Por acelerador | Server consultas/s | p99 Server |
|---|---|---|---|---|---|
| 8× H200 SXM 141 GB | NVIDIA | 756960 | 94620 | 632229 | 12.98 ms |
| 8× H100 SXM 80 GB | Dell XE9680 | 709849 | 88731 | 584207 | 10.39 ms |
| 8× H200 MaxQ, con potencia medida | NVIDIA | 556234 | 69529 | 480131 | 7.17 ms |
| 1× speedAI240 Slim | UntetherAI | 56277 | 56277 | no enviado | p99 SingleStream 0.123 ms |

### BERT-Large 99%, SQuAD v1.1

| Sistema | Offline muestras/s | Server consultas/s | p99 Server |
|---|---|---|---|
| 8× H200 SXM | 73310 | 57609 | 80.79 ms |
| 8× H100 SXM, Dell XE9680 | 70594 | 56012 | 76.26 ms |
| 8× H200 MaxQ | 54063 | 41599 | 118.68 ms |

### Stable Diffusion XL

| Sistema | Offline muestras/s | Por acelerador | Server consultas/s |
|---|---|---|---|
| 8× H200 SXM | 17.42 | 2.18 | 16.78 |
| 8× H100 SXM, DGX H100 | 16.35 | 2.04 | 15.72 |
| 4× TPU v6e Trillium | 5.44 | 1.36 | 4.49 |
| 4× TPU v5e | 1.75 | 0.44 | 1.55 |
| 1× Jetson AGX Orin | 0.102 | 0.102 | p99 SingleStream 9.97 s |

Fuente de las tres tablas, MLCommons (2024), entradas Closed de NVIDIA, Dell, Google y UntetherAI. Los números de H200 coinciden con los que publicó NVIDIA (2024).

### Eficiencia energética

Solo NVIDIA y UntetherAI enviaron resultados con potencia en estos modelos. Promediando el medidor del sistema MaxQ salió esto.

| Modelo | Muestras/s | Potencia media del sistema | Muestras/s por vatio |
|---|---|---|---|
| ResNet-50 | 556234 | 4638 W | 119.9 |
| BERT-99 | 54063 | 5228 W | 10.3 |
| SDXL | 13.20 | 4490 W | 0.0029 |

**Quién lidera y por qué.** En throughput la H200 gana los tres modelos, y frente a la H100 la diferencia es pequeña, 6.6% en ResNet-50, 3.8% en BERT y 6.5% en SDXL. Las dos tienen el mismo chip Hopper, lo que cambia es la memoria, 141 GB de HBM3e contra 80 GB, así que en modelos que caben de sobra la ganancia es poca. En SDXL por chip la H200 da 2.18 muestras/s contra 1.36 de la Trillium, 1.6 veces más, y Byrne (2024) reporta lo mismo a nivel de sistema. Lo interesante es que la Trillium mejoró 3.1 veces sobre la v5e por chip, y eso cuadra con la MXU cuatro veces más grande. En eficiencia el dato más llamativo es el de UntetherAI, que con una sola tarjeta de 75 W saca 56277 muestras/s en ResNet-50, más o menos el 60% de una H200, y Byrne (2024) dice que lo hace con cerca del 20% de la potencia, porque su arquitectura pone el cómputo al lado de la memoria y casi no mueve datos. La versión MaxQ de NVIDIA bajó el throughput un 26% para ahorrar energía, eso muestra que el punto de operación de la GPU se puede mover entre rendimiento y vatios. El Jetson Orin aparece en SDXL con casi 10 s por imagen, sirve para ver que en el borde un modelo generativo grande todavía no es práctico.

## 3. Análisis costo-rendimiento y TCO

| Acelerador | Precio consultado | Potencia | Fuente del precio |
|---|---|---|---|
| NVIDIA H100 SXM | cerca de 30000 USD | 700 W | valor de referencia de la guía, NVIDIA no publica precio de lista |
| TPU v6e Trillium | 2.70 USD por chip-hora bajo demanda, 1.22 USD con compromiso de 3 años | no aplica, se paga por hora | Google Cloud (n.d.-c), us-east1 |
| TPU v5e | 1.20 USD por chip-hora bajo demanda | no aplica | Google Cloud (n.d.-c), us-central1 |
| AMD Alveo U55C | 4747 USD | 115 W típica | AMD (n.d.), tienda oficial |
| Mac Studio M3 Ultra | desde 3999 USD | SoC completo | Apple (2025) |

La guía da 8000 USD para la Alveo U55C, pero en la tienda de AMD hoy sale a 4747 USD. En los cálculos se usó el precio consultado.

Para la energía se asumió 0.12 USD por kWh, operación 24 horas los 365 días y 3 años, es decir 26280 horas. Estas suposiciones son del autor y sirven para comparar, no son datos de un proveedor.

### Escenario a, startup con inferencia en la nube a menos de 100 solicitudes por segundo

Con 100 solicitudes por segundo de un modelo de visión tipo ResNet-50 una sola H100 está casi vacía, en MLPerf cada H100 aguanta unas 73000 consultas por segundo con p99 bajo 15 ms, o sea que la carga usaría menos del 0.2% del chip. Comprar la GPU sería pagar por capacidad que no se usa.

| Opción a 3 años | Adquisición | Energía | Total aproximado |
|---|---|---|---|
| Comprar 1 H100 | 30000 USD más el servidor | 700 W × 26280 h = 18396 kWh, unos 2208 USD | más de 32208 USD sin contar servidor ni operación |
| 1 chip TPU v6e bajo demanda | 0 | incluida | 2.70 × 26280 = 70956 USD |
| 1 chip TPU v6e con compromiso de 3 años | 0 | incluida | 1.22 × 26280 = 32062 USD |
| 1 chip TPU v5e bajo demanda solo 12 horas al día | 0 | incluida | 1.20 × 13140 = 15768 USD |

Haciendo la cuenta se ve que comprar no gana claro ni siquiera contra el compromiso de 3 años, porque al precio de la H100 hay que sumarle el servidor, la refrigeración y alguien que lo mantenga. Para una startup con carga baja y que puede cambiar, lo mejor es alquilar en la nube un acelerador pequeño, una TPU v5e o una GPU de inferencia como la L4, y apagarlo cuando no hay tráfico. El costo de software también pesa, con GPU se usa TensorRT o vLLM y casi todo el ecosistema está listo, con TPU hay que pasar el modelo por XLA.

### Escenario b, inferencia en el borde

En una cámara, un celular o un carro el límite no es el dinero sino los vatios y el espacio. Una Alveo U55C gasta 115 W típicos, en 3 años son 3022 kWh y unos 363 USD de energía por equipo más los 4747 USD de la tarjeta, y además necesita un servidor con PCIe al lado. Una NPU integrada ya viene pagada dentro del SoC y gasta pocos vatios, a 5 W serían 131 kWh y unos 16 USD en 3 años. Para celulares y cámaras inteligentes la NPU es la opción que tiene sentido, para un vehículo autónomo que tiene que correr varios modelos a la vez se usa un SoC tipo Jetson Orin con GPU integrada, y la FPGA queda para cuando hay que garantizar latencia en una línea industrial con un servidor cerca, como se ve en el Perfil B.

## 4. Recomendación por perfil

### Perfil A, laboratorio universitario con 10000 USD para visión y NLP hasta 7B parámetros

Se recomienda una estación de trabajo con GPU NVIDIA de 24 a 48 GB de VRAM. Primero, por presupuesto, una H100 de 30000 USD no entra y una estación con GPU de escritorio sí. Segundo, por memoria, un modelo de 7B en BF16 ocupa unos 14 GB solo de pesos, entonces cabe para inferencia, y para ajuste fino se usa LoRA o QLoRA, porque un entrenamiento completo con Adam necesita cerca de 16 bytes por parámetro, unos 112 GB. Tercero, por ecosistema, casi todo el código de investigación sale primero en PyTorch con CUDA, y la TPU obliga a pasar por XLA. La limitación es que no alcanza para preentrenar modelos grandes y que para corridas grandes habría que pedir créditos de nube. Un Mac Studio M3 Ultra de 3999 USD tiene más memoria unificada, pero su Neural Engine es para inferencia con Core ML y el entrenamiento en GPU de Apple tiene menos soporte en las librerías.

### Perfil B, manufactura con inspección visual bajo 5 ms en 50 líneas

Se recomienda una FPGA por línea, el detalle está en la decisión técnica del Paso 6. Los criterios son latencia determinística, que una falla no tumbe las 50 líneas y que el modelo de inspección cambia poco, entonces el costo de diseño en HLS se paga una sola vez. La limitación es el costo de capital y que cada cambio de modelo obliga a volver a sintetizar el diseño.

### Perfil C, proveedor de nube que entrena modelos de 70B en clústeres de 512 aceleradores

Se recomienda un clúster de GPU H100 o H200 con NVLink dentro de cada nodo y InfiniBand entre nodos. El primer criterio es la comunicación colectiva, con 900 GB/s de NVLink por GPU y la librería NCCL el all-reduce de los gradientes se hace rápido dentro del nodo. El segundo es que un proveedor de nube puede comprar GPU, mientras que las TPU solo existen dentro de Google Cloud, entonces no puede montar su propio pod de TPU. El tercero es el rendimiento, en MLPerf v4.1 la H200 da 34864 tokens/s en Llama 2 70B con 8 GPU (NVIDIA, 2024). La limitación es la energía, 512 GPU a 700 W son 358 kW solo en aceleradores, y el precio. Si el proveedor fuera Google, un pod de TPU con ICI y conmutadores ópticos sería igual de válido (Jouppi et al., 2023).
