# Implementación Gradient Descent sin el uso de un framework
# Autor: Ameyalli Contreras Sánchez - A01749075
# Fecha: 24/08/2025
#-----------------------------------------------------------------------------------------------------------------------------
#                                         LIBRERÍAS
#-----------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report

#-----------------------------------------------------------------------------------------------------------------------------
#                                  DATASET Y PREPROCESAMIENTO
#-----------------------------------------------------------------------------------------------------------------------------

# Cargar dataset
data = pd.read_csv("Student_Performance.csv")

# Cambio de tipo de variables
data['Extracurricular Activities'].unique()

# >>> Transformar variables categóricas a numéricas con mapeo
data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'No': 0, 'Yes': 1})


# Cambio de la variable objetivo

# >>> Transformar variable objetivo en categórica failed y passed
data['Performance Index'] = ['Passed' if x > 70 else 'Failed' for x in data['Performance Index']]

# >>> Transformar variable objetivo a binario
data['Performance Index'] = [1 if x == 'Passed' else 0 for x in data['Performance Index']]

# >>> Convertir predicciones a enteros
data['Performance Index'] = data['Performance Index'].astype(int)


# Llenado de valores faltantes

# >>> Fill NaNs con la media de la columna
data.fillna(data['Extracurricular Activities'].median(), inplace=True)

# Balance de clases

# >>> Revisar si las clases 0 y 1 están balanceadas
#data['Performance Index'].value_counts()

#-----------------------------------------------------------------------------------------------------------------------------
# SUBMUESTREO
#-----------------------------------------------------------------------------------------------------------------------------

# Debido al desbalance del dataset, se opta por hacer un submuestreo de la clase mayoritaria
# Contar ejemplos por clase
counts = data['Performance Index'].value_counts()
minority_class = counts.idxmin()
majority_class = counts.idxmax()

# Separar clases
minority_df = data[data['Performance Index'] == minority_class]
majority_df = data[data['Performance Index'] == majority_class].sample(n=len(minority_df), random_state=42)

# Unir y mezclar
sub_balanced_data = pd.concat([minority_df, majority_df]).sample(frac=1, random_state=42).reset_index(drop=True)
# >>> Revisar balanceo
#sub_balanced_data['Performance Index'].value_counts()

#-----------------------------------------------------------------------------------------------------------------------------
# SOBREMUESTREO
#-----------------------------------------------------------------------------------------------------------------------------

# Debido al desbalance de las clases, se balancea el dataset con un sobremuestreo de la clase minoritaria
# Contar ejemplos por clase
counts = data['Performance Index'].value_counts()
minority_class = counts.idxmin()
majority_class = counts.idxmax()

# Separar clases
minority_df = data[data['Performance Index'] == minority_class]
majority_df = data[data['Performance Index'] == majority_class]

# Sobremuestreo
minority_oversampled = minority_df.sample(n=len(majority_df), replace=True, random_state=42)

# Unir y mezclar
sobre_balanced_data = pd.concat([majority_df, minority_oversampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# Revisar balanceo
#sobre_balanced_data['Performance Index'].value_counts()

#-----------------------------------------------------------------------------------------------------------------------------
#                                         FUNCTIONS
#-----------------------------------------------------------------------------------------------------------------------------

# Split Data
def split(data):
    """
    Split the data into features and target randomly.
    Args:
        data: The input data.

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Mezclar los índices aleatoriamente
    indices = np.random.permutation(len(data))
    train_size = int(0.8 * len(data))
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    X = data.drop("Performance Index", axis=1)
    y = data["Performance Index"]

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    return X_train, X_test, y_train, y_test

# Evaluation
def evaluate_model(X, w):
    """
    Train the model using gradient descent.
    X: Training features
    w: Initial weights
    """
    eval_xw = X * w
    predictions = np.sum(eval_xw, axis=1).round()

    for i in range(len(predictions)):
        if predictions.iloc[i] < 0:
            predictions.iloc[i] = 0
        elif predictions.iloc[i] > 100:
            predictions.iloc[i] = 100

    # Transformar predictions en categórica
    predictions = ['1' if x > 70 else '0' for x in predictions]
    predictions = pd.Series(predictions)
    predictions = predictions.astype(int)

    return predictions

# Backpropagation: Gradient Descent
def gradient_descent(X, y, pred, w, w_change, alpha):
    """
    Perform gradient descent optimization.
    X: Training features
    y: Training target
    pred: Predictions
    w: Weights
    w_change: Change in weights
    alpha: Learning rate
    """
    for i in range(len(w)):
        # Calculo del error
        error = y - pred
        grad = np.dot(error, X[:, i])  # Producto punto entre error y columna i

        # Calcular nuevos cambios en los pesos
        w_change[i] = w_change[i] + alpha * grad

        # Actualizar el cambio de peso usando la columna i de X
        w[i] += w_change[i]

    return w, w_change

# Model Performance
def performance(y_true, y_pred):
    """
    Calculate and display performance metrics.
    y_true: True labels
    y_pred: Predicted labels
    """
    # Matriz de confusión y_true vs y_pred
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="magma", xticklabels=['Failed', 'Passed'], yticklabels=['Failed', 'Passed'])

    # Reporte de clasificación
    print(classification_report(y_true, y_pred))

#-----------------------------------------------------------------------------------------------------------------------------
#                                         MODELOS
#-----------------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------------
# MODELO NO BALANCEADO
#-----------------------------------------------------------------------------------------------------------------------------

# Dataset Original
X_train, X_test, y_train, y_test = split(data)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
datos_train = pd.concat([X_train, y_train], axis=1)
datos_test = pd.concat([X_test, y_test], axis=1)

# Inicialización de pesos entre -0.5 y 0.5 
weights = [0.4, 0.8, 0.1, 0.6, 0.3]
# Inicialización de cambios de pesos en cero
w_change = np.zeros(len(weights))

epochs = 150
alpha = 0.001

for epoch in range(epochs):
    # Calcular predicciones
    predictions = evaluate_model(X_train, weights)

    # Actualizar pesos
    weights, w_change = gradient_descent(X_train.values, y_train.values, predictions.values, weights, w_change, alpha)

print("PESOS FINALES:", "\n w1 = ", weights[0], 
                        "\n w2 = ", weights[1], 
                        "\n w3 = ", weights[2], 
                        "\n w4 = ", weights[3], 
                        "\n w5 = ", weights[4])

weights = np.array(weights, dtype=float)

# Evaluar modelo final en conjunto de prueba
test_predictions = evaluate_model(X_test, weights)

# Evaluar performance del modelo
performance(y_test, test_predictions)


#-----------------------------------------------------------------------------------------------------------------------------
# MODELO SUBMUESTREO
#-----------------------------------------------------------------------------------------------------------------------------

### **Dataset submuestreo**
X_train_sub, X_test_sub, y_train_sub, y_test_sub = split(sub_balanced_data)
#print(X_train_sub.shape, X_test_sub.shape, y_train_sub.shape, y_test_sub.shape)
datos_train_sub = pd.concat([X_train_sub, y_train_sub], axis=1)
datos_test_sub = pd.concat([X_test_sub, y_test_sub], axis=1)

# Inicialización de pesos entre -0.5 y 0.5 
weights = [0.4, 0.8, 0.1, 0.6, 0.3]
# Inicialización de cambios de pesos en cero
w_change = np.zeros(len(weights))

epochs = 300
alpha = 0.001

for epoch in range(epochs):
    # Calcular predicciones
    predictions = evaluate_model(X_train_sub, weights)

    # Actualizar pesos
    weights, w_change = gradient_descent(X_train_sub.values, y_train_sub.values, predictions.values, weights, w_change, alpha)

print("PESOS FINALES:", "\n w1 = ", weights[0], 
                        "\n w2 = ", weights[1], 
                        "\n w3 = ", weights[2], 
                        "\n w4 = ", weights[3], 
                        "\n w5 = ", weights[4])

weights = np.array(weights, dtype=float)

# Evaluar modelo final en conjunto de prueba
test_predictions = evaluate_model(X_test_sub, weights)

# Evaluar performance del modelo
performance(y_test_sub, test_predictions)


#-----------------------------------------------------------------------------------------------------------------------------
# MODELO SOBREMUESTREO
#-----------------------------------------------------------------------------------------------------------------------------

### **Dataset sobremuestreo**
X_train_sobre, X_test_sobre, y_train_sobre, y_test_sobre = split(sobre_balanced_data)
#print(X_train_sobre.shape, X_test_sobre.shape, y_train_sobre.shape, y_test_sobre.shape)
datos_train_sobre = pd.concat([X_train_sobre, y_train_sobre], axis=1)
datos_test_sobre = pd.concat([X_test_sobre, y_test_sobre], axis=1)

# Inicialización de pesos entre -0.5 y 0.5 
weights = [0.4, 0.8, 0.1, 0.6, 0.3]
# Inicialización de cambios de pesos en cero
w_change = np.zeros(len(weights))

epochs = 300
alpha = 0.001

for epoch in range(epochs):
    # Calcular predicciones
    predictions = evaluate_model(X_train_sobre, weights)

    # Actualizar pesos
    weights, w_change = gradient_descent(X_train_sobre.values, y_train_sobre.values, predictions.values, weights, w_change, alpha)

print("PESOS FINALES:", "\n w1 = ", weights[0], 
                        "\n w2 = ", weights[1], 
                        "\n w3 = ", weights[2], 
                        "\n w4 = ", weights[3], 
                        "\n w5 = ", weights[4])

weights = np.array(weights, dtype=float)

# Evaluar modelo final en conjunto de prueba
test_predictions = evaluate_model(X_test_sobre, weights)

# Evaluar performance del modelo
performance(y_test_sobre, test_predictions)