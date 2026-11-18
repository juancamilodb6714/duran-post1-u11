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

## 5. Cuantización INT8 frente a FP32

| Precisión | Throughput pico en Tensor Core | Modelo | Exactitud reportada | Restricción de MLPerf |
|---|---|---|---|---|
| FP32 referencia | 67 TFLOPS FP32 en H100, sin Tensor Core | ResNet-50 v1.5 | 76.46% top-1 referencia MLPerf, 76.16% en Wu et al. | referencia del 100% |
| FP16 o BF16 | T4 65 TFLOPS, A100 312 TFLOPS, H100 989 TFLOPS | ResNet-50 v1.5 | prácticamente igual a FP32 | al menos 99% de FP32, 75.70% |
| INT8 calibrado | T4 130 TOPS, A100 624 TOPS, H100 1979 TOPS | ResNet-50 v1.5 | 76.14% con el mejor método, 76.05% con entropía | al menos 99% de FP32, 75.70% |
| INT8 calibrado | igual que arriba | BERT-Large, SQuAD v1.1 | F1 90.94 con el mejor método frente a 91.01 en FP32 | al menos 99% o 99.9% de F1 90.874 |

Fuentes, Wu et al. (2020) para las exactitudes, Reddi et al. (2020) y MLCommons (2024) para las restricciones, y las hojas de datos de NVIDIA para los TOPS.

Al poner las tres generaciones lado a lado se ve que la proporción es siempre la misma, INT8 da el doble que FP16 en T4, A100 y H100. Eso pasa porque un multiplicador de 8 bits ocupa mucho menos silicio que uno de 16, y en el mismo Tensor Core caben el doble de operaciones por ciclo. También se mueven la mitad de bytes por peso, lo que ayuda cuando el modelo está limitado por memoria.

Lo que más sorprendió es lo que pasa sin calibrar bien. En la tabla de Wu et al. (2020), BERT-Large con calibración por máximo cae de 91.01 a 85.92 de F1, y con entropía se va a 37.40, mientras que con percentil 99.99% queda en 90.20. EfficientNet b0 con máximo cae de 76.85% a 22.3%. O sea que convertir a INT8 no es un cast, cada capa necesita su propio factor de escala elegido con datos reales, porque si un solo valor atípico define el rango los demás valores quedan amontonados en pocos niveles de los 256 que tiene INT8. Por eso MLPerf exige que todo envío cuantizado conserve el 99% de la exactitud FP32, 99.9% en la variante estricta de BERT, medido sobre el conjunto de validación completo (Reddi et al., 2020). Los envíos de H200 en BERT-99 de las tablas anteriores cumplen esa regla.

## 6. Decisión técnica — ¿GPU centralizada o FPGA distribuida para el Perfil B?

Primero se hizo la cuenta de carga. Son 50 líneas a 10 imágenes por segundo, 500 imágenes por segundo en total. Si la H100 atendiera una imagen a la vez con 1.5 ms por imagen, estaría ocupada 0.75 s de cada segundo, un 75%, y con una cola M/D/1 la espera media sería de unos 2.25 ms, en total cerca de 3.75 ms, todavía bajo los 5 ms en promedio pero con colas largas en los picos. En la práctica la GPU atiende muchas imágenes en paralelo con varios streams o batch pequeño, y en MLPerf una H100 aguanta unas 73000 consultas por segundo de ResNet-50 con p99 bajo 15 ms, así que 500 por segundo es menos del 1% de su capacidad y el throughput sí alcanza. En capital la GPU gana fácil, un servidor con una H100 cuesta unos 30000 USD contra 400000 USD de 50 Alveo al precio de la guía, o 237350 USD al precio actual de AMD. El problema es que la centralización crea un punto único de falla, si se cae la GPU, el servidor o la red se quedan las 50 líneas sin inspección al mismo tiempo, y además las líneas compiten por el mismo dispositivo y el jitter del sistema operativo, del driver y del planificador de CUDA hace que la latencia máxima no esté garantizada aunque el promedio sea bueno. La FPGA corre su pipeline fijo en hardware y la latencia sale igual en cada imagen. Si el rechazo de piezas tiene que ver con seguridad industrial se debe elegir la FPGA distribuida aunque cueste más, porque lo que importa es la latencia máxima y no la media. Si fuera solo control de calidad sin riesgo, la GPU centralizada con un segundo servidor de respaldo sería una opción razonable y mucho más barata.

## 7. Decisión técnica — ¿Conviene cuantizar el modelo del Perfil B a INT8?

Si el modelo ya cumple los 5 ms en FP16 parece que cuantizar no sirve de nada, pero no es así. Con INT8 el mismo hardware hace el doble de inferencias, entonces queda margen de capacidad, en la GPU centralizada eso deja atender más líneas sin subir la latencia de cada una, y en la FPGA el pipeline con multiplicadores de 8 bits gasta menos DSP y deja espacio para un modelo más grande o para más cámaras. Eso es útil si la planta piensa crecer más allá de las 50 líneas. El riesgo está en cómo se valida. Según Wu et al. (2020) ResNet-50 pierde apenas 0.02 puntos de top-1, pero ese número es un promedio sobre 1000 clases de ImageNet, y en una planta la clase defecto es minoritaria, tiene pocas muestras y suele depender de detalles finos como una grieta pequeña, justo lo que más sufre con el ruido de cuantización. Un falso negativo deja pasar una pieza mala, cuesta mucho más que un falso positivo que solo manda una pieza buena a revisión manual. Por eso no basta con el top-1 general, el equipo debe medir precisión y sobre todo recall de la clase defecto con un conjunto de validación propio de la planta, comparando FP16 contra INT8 y usando calibración por percentil o QAT si hace falta. La recomendación es cuantizar a INT8 solo si esa validación confirma que el recall de piezas defectuosas no baja de forma significativa, si baja se deja el modelo en FP16, porque el margen ganado no compensa dejar pasar defectos.

## Conclusiones

No hay un acelerador mejor para todo. La GPU gana en throughput y en flexibilidad, por eso sigue siendo la base para entrenar y para la nube. La TPU es muy competitiva por chip y se alquila barato por hora, pero solo vive en Google Cloud y obliga a usar XLA. La NPU es la única que cabe en un celular o una cámara con pocos vatios. La FPGA es la que da latencia determinística y por eso es la indicada cuando una falla de tiempo tiene consecuencias reales. Revisando los datos de MLPerf se notó también que comparar TOPS de hoja de datos engaña, porque vienen con sparsity o en INT8, y que la cuantización a INT8 duplica el rendimiento de verdad, pero solo con calibración y validando la clase que más importa.

## Referencias

AMD. (n.d.). *AMD Alveo U55C high performance compute card*. Recuperado el 18 de noviembre de 2026, de https://www.amd.com/en/products/accelerators/alveo/u55c/a-u55c-p00g-pq-g.html

Apple. (2025, 5 de marzo). *Apple reveals M3 Ultra, taking Apple silicon to a new extreme* [Comunicado de prensa]. https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/

Byrne, J. (2024, 20 de septiembre). *MLPerf reveals Blackwell gains, Untether's power efficiency*. XPU.pub. https://xpu.pub/2024/09/20/mlperf-4-1/

Google Cloud. (n.d.-a). *TPU v6e*. Recuperado el 18 de noviembre de 2026, de https://docs.cloud.google.com/tpu/docs/v6e

Google Cloud. (n.d.-b). *TPU v5e*. Recuperado el 18 de noviembre de 2026, de https://docs.cloud.google.com/tpu/docs/v5e

Google Cloud. (n.d.-c). *Precios de TPU*. Recuperado el 18 de noviembre de 2026, de https://cloud.google.com/tpu/pricing

Jouppi, N. P., Kurian, G., Li, S., Ma, P., Nagarajan, R., Nai, L., Patil, N., Subramanian, S., Swing, A., Towles, B., Young, C., Zhou, X., Zhou, Z., & Patterson, D. (2023). TPU v4: An optically reconfigurable supercomputer for machine learning with hardware support for embeddings. En *Proceedings of the 50th Annual International Symposium on Computer Architecture* (pp. 1–14). ACM. https://doi.org/10.1145/3579371.3589350

MLCommons. (2024). *MLPerf Inference v4.1 results* [Conjunto de datos]. GitHub. https://github.com/mlcommons/inference_results_v4.1

NVIDIA. (n.d.). *NVIDIA H100 Tensor Core GPU*. Recuperado el 18 de noviembre de 2026, de https://www.nvidia.com/en-us/data-center/h100/

NVIDIA. (2024, 28 de agosto). *NVIDIA Blackwell platform sets new LLM inference records in MLPerf Inference v4.1*. NVIDIA Technical Blog. https://developer.nvidia.com/blog/nvidia-blackwell-platform-sets-new-llm-inference-records-in-mlperf-inference-v4-1/

Reddi, V. J., Cheng, C., Kanter, D., Mattson, P., Schmuelling, G., Wu, C.-J., Anderson, B., Breughe, M., Charlebois, M., Chou, W., Chukka, R., Coleman, C., Davis, S., Deng, P., Diamos, G., Duke, J., Fick, D., Gardner, J. S., Hubara, I., … Zhou, Y. (2020). MLPerf inference benchmark. En *2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture* (pp. 446–459). IEEE. https://doi.org/10.1109/ISCA45697.2020.00045

Wu, H., Judd, P., Zhang, X., Isaev, M., & Micikevicius, P. (2020). *Integer quantization for deep learning inference: Principles and empirical evaluation* (arXiv:2004.09602). arXiv. https://doi.org/10.48550/arXiv.2004.09602
