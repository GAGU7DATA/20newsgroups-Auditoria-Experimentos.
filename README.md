# 20newsgroups: Auditoría de Experimentos y Análisis Crítico

Este repositorio contiene el desarrollo, evaluación y auditoría técnica de modelos de procesamiento de lenguaje natural (NLP) aplicados al dataset clásico **20newsgroups**. El objetivo principal del proyecto es contrastar aproximaciones basadas en arquitecturas profundas frente a métodos probabilísticos tradicionales, evaluando su capacidad de generalización y estabilidad numérica.

## 🚀 Estructura del Proyecto

* `experimentos_auditoria.ipynb`: Jupyter Notebook principal con la carga de datos, preprocesamiento, entrenamiento de redes neuronales, evaluación de hiperparámetros y la auditoría bayesiana.
* `impacto_dimensiones_embedding.png`: Análisis visual del impacto del tamaño del espacio latente en la capa de Embedding.
* `comparativa_performance.png`: Gráfico ejecutivo que contrasta la evolución de las funciones de pérdida y la convergencia del modelo con regularización.

---

## 📊 Núcleo de los Experimentos

### 1. Modelado Profundo (Matriz de Embedding + MLP)
Se implementó una red neuronal profunda con una capa de `Embedding` densa para proyectar los textos en un espacio continuo hiperdimensional, evaluando estratégicamente dos componentes críticos:

* **Regularización (Dropout):** * **Dropout 0.0:** Evidenció un problema severo de sobreajuste (*overfitting*) con una brecha crítica de alta varianza ($F1 \approx 0.99$ en entrenamiento frente a $F1 \approx 0.65$ en validación).
  * **Dropout 0.4 (Punto Óptimo):** Desactivó aleatoriamente el 40% de las neuronas por *forward pass*, forzando representaciones robustas y estabilizando el rendimiento de validación en un $F1 \approx 0.80$.
* **Dimensiones del Espacio Latente:** Se auditó cómo varía la capacidad de la red según el tamaño del vector de embedding, identificando el equilibrio óptimo entre representatividad semántica y costo computacional.

### 2. Auditoría Bayesiana: El Impacto del Suavizado de Laplace ($\alpha$)
Como contrapartida teórica, se analizó un clasificador de **Naive Bayes Multinomial** basado en la hipótesis de independencia condicional. Se evaluó el impacto directo del parámetro de suavizado en la estimación de verosimilitudes:

$$P(t \mid c_k) = \frac{N_{k, t} + \alpha}{N_k + \alpha \cdot |V|}$$

* **Con Suavizado ($\alpha = 1.0$):** El modelo mitiga el riesgo de frecuencias cero asignando una masa mínima de probabilidad a términos no observados en el entrenamiento, alcanzando un **Accuracy sólido de 0.9042**.
* **Sin Suavizado ($\alpha = 0.0$):** La presencia de una sola palabra nueva en el conjunto de testeo propaga un efecto de anulación total en la cadena multiplicativa del teorema ($P(X \mid c_k) = 0$). Esto provocó un desplome neto del rendimiento, cayendo drásticamente a un **Accuracy de 0.6983**.

---

## 🛠️ Tecnologías Utilizadas

* **Python 3.13**
* **Scikit-Learn** (Vectorización TF-IDF y Naive Bayes)
* **Keras / TensorFlow** (Capas de Embedding, Dropout y Densas)
* **Matplotlib & Seaborn** (Visualización corporativa de métricas)

---

## ✒️ Autor
* **Santiago Gabriel Vallejo** - Estudiante de Ciencia de Datos - Universidad Nacional de Almirante Brown (UNaB).