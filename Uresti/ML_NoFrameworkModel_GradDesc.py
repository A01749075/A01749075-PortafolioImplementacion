# Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework.
# GRADIENT DESCENT

# -------------------- LIBRERÍAS ----------------------------
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report

# -------------------- CARGA DE DATOS -----------------------
# Cargar dataset
data = pd.read_csv("Student_Performance.csv")


# -------------------- PREPROCESAMIENTO ---------------------
# Transformar variables categóricas a numéricas con mapeo
data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'No': 0, 'Yes': 1})

# Transformar variable objetivo en categórica failed y passed
data['Performance Index'] = ['Passed' if x > 70 else 'Failed' for x in data['Performance Index']]

# Llenar NaNs con la mediana de la columna
data.fillna(data['Extracurricular Activities'].median(), inplace=True)


# -------------------- DATA SPLIT ---------------------------
# Split data into features and target
X = data.drop("Performance Index", axis=1)
y = data["Performance Index"]

# Split data into training and testing sets sin usar sklearn
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Mostrar conteo del data
print(">>>>>>>>>> CANTIDAD DE DATOS POR SET <<<<<<<<<<")
print("  >>> TRAINING SET:")
print("      X_train tiene: ",len(X_train), " datos")
print("      y_train tiene: ",len(y_train), " datos")

print("  >>> TEST SET:")
print("      X_test tiene: ",len(X_test), " datos")
print("      y_test tiene: ",len(y_test), " datos\n\n\n")


# -------------------- INICIALIZACIÓN -----------------------
# Inicialización de pesos entre -0.5 y 0.5 
weights = [0.1, 0.8, 0.2, 0.6, 0.3]


# -------------------- ENTRENAMIENTO ------------------------
vals_xw = X_train * weights
predictions_train = np.sum(vals_xw, axis=1).round()
for i in range(len(predictions_train)):
    if predictions_train[i] < 0:
        predictions_train[i] = 0
    elif predictions_train[i] > 100:
        predictions_train[i] = 100

# Transformar predictions_train en categórica
predictions_train = ['Passed' if x > 70 else 'Failed' for x in predictions_train]
predictions_train = pd.Series(predictions_train)

# Matriz de confusión 
cm = confusion_matrix(y_train, predictions_train)
sns.heatmap(cm, annot=True, fmt="d", cmap="magma", xticklabels=['Failed', 'Passed'], yticklabels=['Failed', 'Passed'])

# Métricas de desempeño
print(classification_report(y_train, predictions_train))


# -------------------- EVALUACIÓN ---------------------------
# Evaluar modelo en conjunto de prueba
vals_xw_test = X_test * weights
predictions_test = np.sum(vals_xw_test, axis=1).round()
predictions_test = ['Passed' if x > 70 else 'Failed' for x in predictions_test]
predictions_test = pd.Series(predictions_test)

# Matriz de confusión
cm = confusion_matrix(y_test, predictions_test)
sns.heatmap(cm, annot=True, fmt="d", cmap="magma", xticklabels=['Failed', 'Passed'], yticklabels=['Failed', 'Passed']) 

# Métricas de desempeño
print(classification_report(y_test, predictions_test))


