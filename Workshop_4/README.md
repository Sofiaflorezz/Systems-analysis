# Workshop 4 – Computational Simulation Report  
**Analysis of Chaos and Non-Linear Dynamics in the MoA Prediction System**

Este repositorio contiene el informe y recursos del **Workshop 4**, enfocado en la simulación computacional y el análisis de dinámicas caóticas en un sistema de predicción de **Mechanisms of Action (MoA)** a partir de datos biológicos de alta dimensión (reto de Kaggle).


## Objetivo del proyecto

Validar empíricamente la arquitectura propuesta en talleres previos, evaluando:

- La **sensibilidad del sistema** a perturbaciones en la distribución de entrada, especialmente en el parámetro `cp_time`.
- La presencia de **comportamientos no lineales y caóticos** que afecten la estabilidad del desempeño del modelo.
- Propuestas de refinamiento arquitectónico para hacer el sistema más robusto y confiable.


## Datos y preprocesamiento

- Filtrado de controles y preparación del dataset de entrenamiento.
- **Estandarización** diferenciada por tipo de feature (genes, células, metadata, etc.).
- **Reducción dimensional** mediante PCA para mitigar ruido y correlaciones fuertes.
- Diseño de escenarios para probar cambios explícitos en la distribución de `cp_time`.


## Escenarios de simulación

### Escenario 1 – Modelo de Machine Learning (Random Forest)

- Modelo **Random Forest multi-salida** entrenado sobre el dataset preprocesado.
- Se aplican **perturbaciones controladas** en la distribución de `cp_time`.
- Se evalúa la estabilidad del desempeño mediante:
  - **Log-loss medio**
  - **Desviación estándar**
  - **Coeficiente de variación (CoV)**

El objetivo es cuantificar qué tan estable es el modelo frente a cambios sutiles en la distribución de entrada.

### Escenario 2 – Autómata Celular (Cellular Automata, CA)

- Implementación de un **autómata celular 2D** que modela la propagación de señales biológicas locales.
- Búsqueda de un **punto de bifurcación** donde el sistema pasa:
  - De un comportamiento estable/predecible  
  - A un régimen de **caos espaciotemporal**, altamente sensible a los parámetros de control (metadata).


## Resultados principales

- El sistema muestra **alta sensibilidad** a perturbaciones en `cp_time`, reflejada en un incremento notable del CoV y la variabilidad del log-loss.
- El autómata celular evidencia la existencia de un umbral de parámetros donde la dinámica se vuelve **caótica**, reforzando la interpretación de que el sistema completo puede entrar en regiones de comportamiento no lineal difícil de controlar.


## Propuestas arquitectónicas

Para robustecer el sistema de predicción MoA, el informe propone:

1. **Módulo activo de validación de distribución**  
   - Detectar cambios en la distribución de entrada antes de hacer predicciones.
   - Disparar alertas o estrategias de recalibración cuando se detecten “shifts” significativos.

2. **Arquitectura Mixture of Experts estratificada por `cp_time`**  
   - Modelos especializados por rango de `cp_time`.
   - Mezcla ponderada de expertos en función de la metadata observada.

3. **Incorporación de incertidumbre bayesiana (p. ej. MC Dropout)**  
   - Reportar junto con cada predicción una **medida explícita de confianza**.
   - Permitir decisiones más informadas en presencia de posibles regiones caóticas del espacio de entrada.


## Contenido del repositorio (sugerido)

- `report/`  
  Informe en PDF del Workshop 4.
- `notebooks/`  
  Notebooks con experimentos de Random Forest y perturbación de `cp_time`.
- `ca_simulation/`  
  Implementación del autómata celular y experimentos de caos.
- `data/`  
  Scripts de preprocesamiento y/o muestras de datos (si aplica).

