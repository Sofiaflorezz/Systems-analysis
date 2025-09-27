
# Proyecto: Predicción de Mecanismos de Acción (MoA)  

Este proyecto se centra en la competencia MoA Prediction, cuyo objetivo es predecir los mecanismos de acción de distintos compuestos a partir de datos biológicos de alta dimensión.  

Se trabaja con datos de expresión génica, viabilidad celular y metadatos experimentales, aplicando técnicas de Machine Learning.

-----

# Archivos del proyecto  

- **train_drug.csv** → Datos de entrenamiento (incluyen la variable objetivo con las etiquetas de fármacos o mecanismos de acción).  
- **test_features.csv** → Datos de prueba (solo variables predictoras, sin la columna objetivo).  
- **sample_submission.csv** → Archivo de ejemplo de cómo deben enviarse las predicciones.  
- **submission.csv** → Archivo final con las predicciones generadas por el modelo.  

-----

# Flujo del sistema 

1. **Entradas (Inputs)**  
   - Datos de expresión génica.  
   - Medidas de viabilidad celular.  
   - Metadatos experimentales.

2. **Preprocesamiento**  
   - Normalización de variables.  
   - Reducción de dimensionalidad (PCA, selección de características).  
   - Manejo de variabilidad y control de ruido.  

3. **Modelado**  
   - Se aplican algoritmos de aprendizaje automático para transformar los datos en predicciones.  
   - Se probaron enfoques (árboles de decisión, random forest)

4. **Evaluación**  
   - Se comparan las probabilidades predichas con las etiquetas reales usando la métrica log loss.  
   - Retroalimentación que permite ajustar y mejorar el modelo.  

5. **Salida (Outputs)**  
   - Archivo submission.csv con las predicciones para cada muestra y mecanismo de acción.  

---

# Código de entrenamiento del modelo  

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Cargar datos de entrenamiento
datos_entrenamiento = pd.read_csv("train_drug.csv")

# Separar variables predictoras y variable objetivo
X_entrenamiento = datos_entrenamiento.drop("Drug", axis=1)
y_entrenamiento = datos_entrenamiento["Drug"]

# Definir y entrenar el modelo
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_entrenamiento, y_entrenamiento)

# Cargar datos de prueba
datos_prueba = pd.read_csv("test_features.csv")

# Realizar predicciones
predicciones = modelo.predict(datos_prueba)

# Guardar resultados en un archivo CSV
pd.DataFrame({"Prediccion": predicciones}).to_csv("submission.csv", index=False)
