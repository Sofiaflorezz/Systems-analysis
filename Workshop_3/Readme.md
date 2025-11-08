# MoA Prediction - Workshop 3

Este proyecto consolida el diseño de un sistema de predicción de Mecanismos de Acción (MoA) aplicando principios de **ingeniería robusta** y **gestión de proyectos**. El objetivo es refinar la arquitectura y elevar su **confiabilidad, escalabilidad, mantenibilidad y usabilidad**, alineando las decisiones con estándares de calidad como **ISO 9000**, **CMMI** y **Six Sigma**.

### Arquitectura y componentes
El sistema se estructura en una **arquitectura modular** con cinco módulos ejecutables e independientes, lo que facilita pruebas, reemplazos y evolución del pipeline:
- **Data Loader:** carga y validación del dataset (verificación de columnas, faltantes).
- **Preprocessing:** limpieza, transformación y normalización de variables.
- **Model Trainer:** entrenamiento y registro de métricas (Random Forest / NN simple).
- **Prediction:** generación de predicciones y archivo de envío.
- **Report Generator:** visualización de métricas y resultados.
La modularidad permite intercambiar modelos/datasets sin afectar el resto del sistema y mantener **versionado** y **reproductibilidad** (GitHub).

### Calidad, riesgos y mitigación
Se identificaron riesgos clave y se definieron estrategias de control:
- **Calidad de datos:** EDA, detección de faltantes/outliers, normalización y alertas tempranas.
- **Sobreajuste:** validación cruzada, ajuste de hiperparámetros, early stopping y versionado de modelos.
- **Interpretabilidad:** uso de **SHAP/LIME** y reportes de importancia de características para validar consistencia biológica.
- **Flujo del pipeline:** controles de entrada/salida por módulo y verificación de persistencia de resultados.

### Gestión del proyecto
Se establecieron **roles**, **hitos** y **entregables** con un cronograma de cinco semanas, y se adoptó **Kanban** (Trello/Miro) para la orquestación del trabajo (visibilidad del estado, adaptación a cambios, detección de bloqueos y seguimiento transparente). La documentación y el control de versiones en **GitHub** aseguran trazabilidad y colaboración.

### Mejoras y estado actual
Durante esta fase se **refinó la arquitectura**, se precisaron **estrategias de mitigación y monitoreo**, y se reforzó la **documentación**. El sistema queda preparado para fases siguientes de implementación y optimización continua, manteniendo calidad de resultados y sostenibilidad del desarrollo.
