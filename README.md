# Red Neuronal para Predicción de Enfermedad Cardíaca

## 📋 Descripción General

Este proyecto implementa una red neuronal profunda (DNN) utilizando TensorFlow/Keras para predecir la presencia de enfermedad cardíaca basándose en características médicas de los pacientes. El modelo es una herramienta de apoyo diagnóstico que analiza múltiples factores clínicos.

## 👤 Autor

![Logo](logogit.png)

- **Autor:** _Eduardo Efrain Garcia Sarez_

  > **27/11/2025**

- **Institución:** **Instituto Tecnológico Superior de Jalisco (TSJ)**.  
- **Carrera:** **Ingeniería en Sistemas Computacionales**

Proyecto desarrollado como parte de actividades de aprendizaje en Deep Learning

---

**Última actualización**: 28 de noviembre de 2025

## 🏥 Dataset

**Fuente**: HeartDiseaseTrain-Test.csv

### Variables Clínicas (13 características de entrada)

| Variable | Descripción | Valores |
|----------|-------------|---------|
| **age** | Edad del paciente | Años |
| **sex** | Sexo del paciente | 1 = masculino, 0 = femenino |
| **cp** | Tipo de dolor torácico | 0-3 (Típica, Atípica, No anginosa, Asintomático) |
| **trestbps** | Presión arterial en reposo | mm Hg |
| **chol** | Colesterol sérico | mg/dl |
| **fbs** | Glucosa en ayunas > 120 mg/dl | 1 = verdadero, 0 = falso |
| **restecg** | Resultados ECG en reposo | 0-2 (Normal, Anormalidad ST-T, Hipertrofia) |
| **thalach** | Frecuencia cardíaca máxima alcanzada | bpm |
| **exang** | Angina inducida por ejercicio | 1 = sí, 0 = no |
| **oldpeak** | Depresión del ST inducida por ejercicio | mm |
| **slope** | Pendiente segmento ST | 0-2 (Ascendente, Plana, Descendente) |
| **ca** | Vasos principales coloreados | 0-3 |
| **thal** | Talasemia | 1-3 (Normal, Defecto fijo, Defecto reversible) |

**Variable Objetivo**: `target` (1 = Enfermedad presente, 0 = Ausencia de enfermedad)

## 🧠 Arquitectura del Modelo

### Arquitectura Neuronal

```psh
Entrada (13 características)
    ↓
Dense(64, ReLU)
    ↓
Dense(32, ReLU)
    ↓
Dense(16, ReLU)
    ↓
Dense(1, Sigmoid) → Salida [0, 1]
```

### Configuración de Entrenamiento

| Parámetro | Valor |
|-----------|-------|
| **Optimizer** | Adam (adaptativo, convergencia rápida) |
| **Loss Function** | Binary Crossentropy |
| **Epochs** | 100 |
| **Batch Size** | 32 |
| **Validation Split** | 10% |
| **Metrics** | Accuracy |

### Justificación Técnica

✅ **Red Neuronal Profunda (DNN)**:

- Captura relaciones no lineales complejas en factores médicos
- Aproximación universal de funciones continuas
- Extracción automática de características jerárquicas
- Supera métodos tradicionales en clasificación médica

✅ **Keras API**:

- Interfaz intuitiva y de alto nivel
- Integración nativa con TensorFlow
- Prototipado rápido y flexible
- Código legible y conciso

## 📊 Evaluación del Modelo

### Métricas Utilizadas

#### 1. **Accuracy (Exactitud)**

```psh
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Proporción de predicciones correctas sobre el total.

#### 2. **Confusion Matrix (Matriz de Confusión)**

|  | Predicción Negativa | Predicción Positiva |
|---|---|---|
| **Real Negativo** | TN (Verdadero Negativo) | FP (Falso Positivo) |
| **Real Positivo** | FN (Falso Negativo) | TP (Verdadero Positivo) |

#### 3. **Métricas Derivadas**

- **Precision**: TP / (TP + FP) - Precisión de predicciones positivas
- **Recall/Sensitivity**: TP / (TP + FN) - Capacidad de detectar enfermos
- **F1-Score**: Media armónica de Precision y Recall

### Interpretación Clínica

⚠️ **En contexto médico**:

- Los **Falsos Negativos (FN) son críticos**: Pacientes enfermos clasificados como sanos
- Los **Falsos Positivos (FP)** son menos graves: Alarmas innecesarias
- Se prioriza **Recall alto** para no pasar por alto enfermedades

## 🚀 Uso del Modelo

### Instalación de Dependencias

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn
```

### Cargar y Entrenar el Modelo

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Cargar dataset
df = pd.read_csv(
    'https://raw.githubusercontent.com/fenago/deep-learning-essentials/main/HeartDiseaseTrain-Test.csv'
)

# Manejo de valores faltantes
df = df.dropna()

# Separar características y variable objetivo
X = df.drop(columns=['target'])
y = df['target'].values

# Codificar variables categóricas con One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

# Codificar variable objetivo
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Dividir datos (10% para prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Construir modelo
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilar
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Entrenar
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Resumen del modelo
model.summary()
```

### Realizar Predicciones

```python
# Predecir probabilidades para muestras de prueba
y_pred_prob = model.predict(X_test, verbose=0)

# Convertir probabilidades a predicciones binarias (umbral 0.5)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Crear DataFrame con resultados
resultados = pd.DataFrame({
    'Predicción': y_pred,
    'Real': y_test
})

# Calcular exactitud
accuracy = np.mean(y_pred == y_test)
print(f"Exactitud sobre conjunto de prueba: {accuracy*100:.2f}%")
print(resultados.head(10))
```

## 📈 Visualización de Resultados

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Calcular matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Visualizar matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Sin Enfermedad', 'Con Enfermedad'],
            yticklabels=['Sin Enfermedad', 'Con Enfermedad'])
plt.ylabel('Valor Real')
plt.xlabel('Predicción')
plt.title('Matriz de Confusión - Predicción de Enfermedad Cardíaca')
plt.show()

# Visualizar curvas de entrenamiento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Pérdida del Modelo')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Exactitud del Modelo')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.show()
```

El modelo genera:

1. **Matriz de Confusión**: Visualización heatmap de TP, TN, FP, FN
2. **Curvas de Entrenamiento**: Pérdida y exactitud en conjuntos de entrenamiento/validación

## ⚠️ Consideraciones Importantes

- ✅ El modelo es una **herramienta de apoyo diagnóstico**
- ❌ **Nunca reemplaza** el juicio clínico de profesionales médicos
- 🔍 Resultados deben ser **validados por especialistas**
- 📊 Requiere **interpretación profesional**
- 📌 **Split de datos**: 90% entrenamiento, 10% prueba
- 📌 **Validación cruzada**: 20% del entrenamiento para validación

## 📁 Archivos del Proyecto

```psh
hands_on_5/
├── on5.ipynb                          # Notebook Jupyter con implementación
├── trash.md                           # Documentación técnica detallada
├── README.md                          # Este archivo
└── HeartDiseaseTrain-Test.csv         # Dataset (si está incluido)
```

## 🔧 Tecnologías Utilizadas

- **Python 3.x**
- **TensorFlow 2.x** - Framework de deep learning
- **Keras** - API de alto nivel
- **Pandas** - Manipulación de datos
- **NumPy** - Operaciones numéricas
- **Scikit-learn** - Métricas y validación
- **Matplotlib & Seaborn** - Visualización

## 📚 Referencias

- [TensorFlow/Keras Documentation](https://www.tensorflow.org/guide/keras)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Neural Networks Theory](https://en.wikipedia.org/wiki/Artificial_neural_network)
- [**“Build Your First Neural Network with TensorFlow: A Step-by-Step Guide”**](https://drlee.io/build-your-first-neural-network-with-tensorflow-a-step-by-step-guide-1dd3e6652cf1)
