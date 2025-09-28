# Predicción de mecanismos de acción (MoA) 

Este repositorio contiene el análisis y la implementación realizada para la competencia **Mechanisms of Action (MoA) Prediction** de Kaggle.

## Análisis del sistema

El análisis del sistema se realizó siguiendo un enfoque de **ingeniería de sistemas**, con el objetivo de comprender y modelar la dinámica de la competencia.

### 1) Definición de componentes
- **Entradas**: variables de expresión génica (*g-variables*), características celulares (*c-variables*) y metadatos experimentales (tipo de compuesto, dosis y tiempo de exposición).
- **Procesos**: preprocesamiento (normalización, reducción de dimensionalidad y selección de variables), modelado mediante algoritmos de clasificación multietiqueta (redes neuronales, *gradient boosting* y regresión logística), y evaluación con la métrica *log loss*.
- **Salidas**: vectores de probabilidad para cada uno de los **206** mecanismos de acción.

### 2) Relaciones y dependencias
Se identificó cómo las variables de entrada interactúan entre sí: la expresión génica influye en la viabilidad celular y los metadatos (tipo de compuesto, dosis, tiempo) **modulan** esas respuestas. Estas dependencias condicionan el rendimiento de los modelos y muestran la **sensibilidad** del sistema a parámetros como *cp_type* o *cp_dose*.

### 3) Flujo de datos y retroalimentación
El sistema se representó en un flujo de módulos:
**Entrada → Preprocesamiento → Modelado → Evaluación → Salida**


### 4) Perspectiva sistémica
- **Limitaciones**: no descubre nuevos mecanismos y depende de la calidad de los datos (clases raras, ruido, efectos de lote).
- **Requisitos no funcionales**: reproducibilidad, control de versiones (datos/modelos) y documentación del pipeline.

## Conclusión
El problema se entiende como un sistema **interconectado, no lineal y sensible**, donde cada módulo depende del anterior. Una coordinación adecuada entre preprocesamiento, modelado y evaluación permite obtener predicciones consistentes y útiles para investigación biomédica.
