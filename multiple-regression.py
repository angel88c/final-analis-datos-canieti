import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


@st.cache_data
def read_csv():
    return pd.read_csv("dataset_clase5.csv")

def tidv_corr_matrix(corr_mat):
    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ["variable_1", "variable_2", "r"]
    corr_mat = corr_mat.loc[corr_mat["variable_1"]!=corr_mat["variable_2"],:]
    corr_mat["abs_r"]=np.abs(corr_mat["r"])
    corr_mat = corr_mat.sort_values("abs_r", ascending=False)
    return corr_mat    
    
if __name__ == '__main__':
    st.title("Session 05 - regression multiple")
    
    df = read_csv()
    corr_matrix = df.select_dtypes(include=["float", "int"]).corr(method="pearson")
    
    st.write(tidv_corr_matrix(corr_matrix))
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4))
    st.pyplot(fig)
    
    sns.heatmap(
    corr_matrix,
    annot=True,
    cbar=False,
    annot_kws={"size":8},
    vmin=-1,
    vmax=1,
    center=0,
    cmap=sns.diverging_palette(20,220,n=200),
    square=True,
    ax=ax
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation = 45,
        horizontalalignment = 'right',
    )

    ax.tick_params(labelsize = 10)
    #Variables independientes y varaible dependiente
    st.write(fig)

    X=df[['HORAS EN REPRODUCCION','VECES']]
    y=df['INGRESO']

    ##Dividir datos en entrenamiento y prueba
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    ##Eentrenar el modelo
    modelo=LinearRegression()
    modelo.fit(X_train,y_train)
    
    #st.write(modelo.coef_[1])
    #st.write(modelo.intercept_)

    ##Predicciones
    y_pred=modelo.predict(X_test)
    y_pred

    #Evaluar el rendimiento de modelo
    r2=r2_score(y_test,y_pred)
    r2
    mse=mean_squared_error(y_test,y_pred)
    mse