# -*- coding: utf-8 -*-
"""
Created on Thu May  9 20:46:17 2024

@author: Clau Lorenzo
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error



# Crear DataFrame
#dataframe = pd.DataFrame({'equipos': equipos, 'bateos': bateos, 'carreras': runs})
dataframe=pd.read_csv("C:\\Users\\Giovanna Velázquez\\OneDrive\Documentos\\Tecmilenio\\PYTHONAnálisis de datosMAY-JUN2024\\SESION 4\\dataset_clase4_regsimple.csv")
#dataframe=pd.read_excel() por si quieren hacerlo con excel

# Visualizar los primeros registros del DataFrame
print(dataframe.head())

# Graficar la distribución de VENTAS VS COSTO
plt.figure(figsize=(8, 6))
plt.scatter(dataframe['Sales'], dataframe['Shipping  Cost'], color='firebrick', label='Datos')
plt.title('Distribución de costo y ventas')
plt.xlabel('Ventas')
plt.ylabel('Costo')
plt.legend()
plt.show()

# Calcular correlación de Pearson
corr, p_value = pearsonr(dataframe['Sales'], dataframe['Shipping  Cost'])
print("Coeficiente de correlación de Pearson:", corr)
print("Valor p:", p_value)

# Realizar regresión lineal
X = dataframe[['Sales']]
y = dataframe['Shipping  Cost']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
y_pred



#### ECUACION EN EL GRAFICO####
coeficiente=model.coef_
intercepto=model.intercept_
print("Eciacion de regresion:")
print(f"y={coeficiente[0]}*X+{intercepto}")



# Coeficientes de la regresión
coeficiente = model.coef_[0]
interseccion = model.intercept_
print("Coeficiente de la pendiente:", coeficiente)
print("Intersección:", interseccion)



# Graficar la regresión lineal
plt.figure(figsize=(8, 6))
plt.scatter(dataframe['Sales'], dataframe['Shipping  Cost'], color='firebrick', label='Datos')
plt.plot(X, y_pred, color='blue', linewidth=2, label='Regresión Lineal')
plt.title('Regresión Lineal Simple')
plt.xlabel('Ventas')
plt.ylabel('Costo')
plt.legend()
plt.show()

# Calcular coeficiente de determinación (R^2) y error cuadrático medio (MSE)
r2 = model.score(X, y)
rmse = mean_squared_error(
        y_true  = y,
        y_pred  = y_pred,
        squared = False
       )

print("Coeficiente de determinación (R^2):", r2)   #INDICA LA PROPORCION DE LA VARIABILIDAD DE LA VARIABLE DEPENDIENTE POR EL MODELO
print("Error cuadrático medio (MSE):", rmse)  ### ENTRE MENOR SEA EL VALOR DEL MSE, MEJOR SERA EL AJUSTE DEL MODELO DE LOS DATOS OBSERVADOS


