# Implementación de un algoritmo de ML usando framework
# Autor: Ameyalli Contreras Sánchez - A01749075
# Fecha: 10/09/2025

#-----------------------------------------------------------------------------------------------------------------------------
#                                         LIBRERÍAS
#-----------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


#-----------------------------------------------------------------------------------------------------------------------------
#                                         FUNCTIONS
#-----------------------------------------------------------------------------------------------------------------------------

# Performance
def performance(X_test, y_test, randomforest):
    y_pred = randomforest.predict(X_test) # predicciones de la clase
    y_proba = randomforest.predict_proba(X_test)[:, 1] # probabilidades de la clase positiva
    
    print("\n>>> REPORTE DE CLASIFICACIÓN: <<<")
    print(classification_report(y_test, y_pred))

    print("\n>>> AUC-ROC:", roc_auc_score(y_test, y_proba), "<<<")

    print("\n>>> MATRIZ DE CONFUSIÓN: <<<")
    c_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(c_matrix, annot=True, fmt="d", cmap="magma", xticklabels=['Failed', 'Passed'], yticklabels=['Failed', 'Passed'])

# Importancia de variables
def importance(X, randomforest, X_test, y_test):
    perm_importance = permutation_importance(randomforest, X_test, y_test, n_repeats=10, random_state=42)

    importances = pd.Series(perm_importance.importances_mean, index=X.columns)
    importances.nlargest(10).plot(kind="barh", color = 'm')
    plt.title("Permutation Importance")
    plt.show()

# Grid Search
def gridSearch(X_train, y_train, randomforest, parameters):
    grid_search = GridSearchCV(
        estimator=randomforest,
        param_grid=parameters,
        scoring="roc_auc",   # puedes cambiar a "f1" según tu objetivo
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)
    print("\n>>> MEJORES PARÁMETROS: <<<\n", grid_search.best_params_)
    return grid_search



#-----------------------------------------------------------------------------------------------------------------------------
#                                  DATASET Y PREPROCESAMIENTO
#-----------------------------------------------------------------------------------------------------------------------------

# Importar datos
data = pd.read_csv('Student_Performance.csv')

# Convertir extracurricular activities en binaria
data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'No': 0, 'Yes': 1})

# >>> Ajuste de la variable objetivo

# Transformar variable objetivo en categórica failed y passed
data['Performance Index'] = ['Passed' if x > 70 else 'Failed' for x in data['Performance Index']]
# Transformar variable objetivo a binario
data['Performance Index'] = [1 if x == 'Passed' else 0 for x in data['Performance Index']]
# Convertir predicciones a enteros
data['Performance Index'] = data['Performance Index'].astype(int)

# Revisar si las clases 0 y 1 están balanceadas
#data['Performance Index'].value_counts()


#-----------------------------------------------------------------------------------------------------------------------------
#                                         DATA SPLIT
#-----------------------------------------------------------------------------------------------------------------------------

# Dividir en variables y target
X = data.drop('Performance Index', axis=1)
y = data['Performance Index']

# Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


#-----------------------------------------------------------------------------------------------------------------------------
#                                         MODELOS
#-----------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------
# MODELO INICIAL
#-----------------------------------------------------------------------------------------------------------------------------

rf = RandomForestClassifier(
    n_estimators=20,          # número de árboles
    max_depth=8,              # profundidad máxima (None = hasta que las hojas sean puras)
    min_samples_split=200,    # mínimo de muestras para dividir un nodo
    min_samples_leaf=100,     # mínimo de muestras por hoja
    max_features="sqrt",      # nº de features consideradas en cada split
    bootstrap=True,           # muestreo con reemplazo
    class_weight="balanced",  # útil en caso de desbalance
    random_state=42,          # reproducibilidad
    n_jobs=-1                 # uso de todos los núcleos
)

# Entrenamiento
rf.fit(X_train, y_train)

# Evaluación de desempeño
performance(X_test, y_test, rf)

# Importancia de variables
importance(X, rf, X_test, y_test)

#-----------------------------------------------------------------------------------------------------------------------------
# MODELOS MEJORADOS CON OPTIMIZACIÓN DE HIPERPARÁMETROS
#-----------------------------------------------------------------------------------------------------------------------------

# Optimización de hiperparámetros con GridSearch
param_grid = {
    "n_estimators": [10, 15, 20, 25, 30, 40],
    "max_depth": [6, 8, 10, 15, 20],
    "min_samples_split": [20, 40, 50, 100, 150, 200],
    "min_samples_leaf": [20, 40, 50, 100],
    "max_features": ["sqrt", "log2"],
}

grid_search_res = gridSearch(X_train, y_train, rf, param_grid)

# NUEVO RANDOM FOREST OPTIMIZADO
best_rf = grid_search_res.best_estimator_
best_rf.fit(X_train, y_train)

#-----------------------------------------------------------------------------------------------------------------------------
# TOMANDO TODAS LAS VARIABLES
#-----------------------------------------------------------------------------------------------------------------------------

# Desempeño
performance(X_test, y_test, best_rf)

# Importancia de variables 
importance(X, best_rf, X_test, y_test)

#-----------------------------------------------------------------------------------------------------------------------------
# TOMANDO LAS 4 VARIABLES MÁS IMPORTANTES
#-----------------------------------------------------------------------------------------------------------------------------

X_train_new2 = X_train.drop(['Sample Question Papers Practiced'], axis =1)
X_test_new2 = X_test.drop(['Sample Question Papers Practiced'], axis =1)

best_rf.fit(X_train_new2, y_train)
performance(X_test_new2, y_test, best_rf)
importance(X.drop(['Sample Question Papers Practiced'], axis =1), best_rf, X_test_new2, y_test)

#-----------------------------------------------------------------------------------------------------------------------------
# TOMANDO LAS 3 VARIABLES MÁS IMPORTANTES
#-----------------------------------------------------------------------------------------------------------------------------

X_train_new1 = X_train.drop(['Sample Question Papers Practiced', 'Extracurricular Activities'], axis =1)
X_test_new1 = X_test.drop(['Sample Question Papers Practiced', 'Extracurricular Activities'], axis =1)

best_rf.fit(X_train_new1, y_train)
performance(X_test_new1, y_test, best_rf)
importance(X.drop(['Sample Question Papers Practiced', 'Extracurricular Activities'], axis =1), best_rf, X_test_new1, y_test)

#-----------------------------------------------------------------------------------------------------------------------------
# TOMANDO LAS 2 VARIABLES MÁS IMPORTANTES
#-----------------------------------------------------------------------------------------------------------------------------

X_train_new = X_train.drop(['Sample Question Papers Practiced', 'Extracurricular Activities', 'Sleep Hours'], axis =1)
X_test_new = X_test.drop(['Sample Question Papers Practiced', 'Extracurricular Activities', 'Sleep Hours'], axis =1)

best_rf.fit(X_train_new, y_train)
performance(X_test_new, y_test, best_rf)
importance(X.drop(['Sample Question Papers Practiced', 'Extracurricular Activities', 'Sleep Hours'], axis =1), best_rf, X_test_new, y_test)