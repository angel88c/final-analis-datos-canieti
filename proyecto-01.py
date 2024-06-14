import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st

import scipy.stats as stats
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
from math import ceil

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error

@st.cache_data
def read_csv():
    return pd.read_csv("insurance.csv")

def tidv_corr_matrix(corr_mat):
    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ["variable_1", "variable_2", "r"]
    corr_mat = corr_mat.loc[corr_mat["variable_1"]
                            != corr_mat["variable_2"], :]
    corr_mat["abs_r"] = np.abs(corr_mat["r"])
    corr_mat = corr_mat.sort_values("abs_r", ascending=False)
    return corr_mat

if __name__ == '__main__':

    st.title("Análisis de póliza de Seguros.")
    
    st.write("""
             El siguiente Proyecto plantea el analisis de regresion lineal para una 
             tabla de poliza de seguros para la predicción del precio de la poliza de 
             acuerdo a las variables edad, ínidce de masa corporal y numero de hijos del
             asegurado.
             """)
    
    df = read_csv()
    
    
    corr_matrix = df.select_dtypes(
        include=["float", "int"]).corr(method="pearson")
    
    st.markdown("## Matriz de Correlacion")
    column1, column2 = st.columns(2)
    with column1:        
        st.write(tidv_corr_matrix(corr_matrix))
    
    
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        #st.pyplot(fig)

        sns.heatmap(
            corr_matrix,
            annot=True,
            cbar=False,
            annot_kws={"size": 8},
            vmin=-1,
            vmax=1,
            center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True,
            ax=ax)
        
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right',
        )
        ax.tick_params(labelsize=10)

       
    
    with column2:
         # Variables independientes y varaible dependiente
        st.write(fig)
        
    st.markdown("## Interpretación.")
    st.write("""
             Para este caso se analizan 3 variables, la edad (age), el numero de hihos (children)
             y el índice de masa corporal (bmi), encontrando que existe una mayor correlacion 
             entre el indice de masa corporal y la edad, no tanto el número de hijos
             """)
    
    
    x = df[["bmi", "children", "age"]]
    y = df[["charges"]]
    
    st.markdown("## Expresión de regresión.")
    st.write("Se obtiene del dataset las variables de analisis x (age, bmi, children) e y (charges)")
    
    
    #split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    #Train the model
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    st.markdown("### Coefficients: ")
    bmi = model.coef_[0][0]
    children = model.coef_[0][1]
    age = model.coef_[0][2]
    
    st.write({
        "bmi": bmi,
        "children": children,
        "age": age
    })
    st.markdown("### Interceptor: ")
    st.write({
        "interceptor": model.intercept_
    })
    
    st.write("Lo cual nos entrega una expresión de:")
    st.latex(rf'''
        y = {bmi}\cdot bmi + {children} \cdot children + {age} \cdot age 
    ''')
    
    #message = "y = "
    #for i in range(len(model.coef_[0])):
    #    message += f'{model.coef_[0][i]} * {x.columns[i]} + '
    #message += f'{model.intercept_[0]}' 
    #st.write(message)
    
    # Predictions
    y_pred = model.predict(x_test)
    
    #Evaluate the model
    r2 = r2_score(y_test, y_pred)
    
    mse = root_mean_squared_error(y_test, y_pred)
    
    st.write("")
    
    st.markdown("### Coeficiente de determinación: ")
    st.write(r2)
    st.markdown("### Error Cuadratico medio: ")
    st.write(mse)
    
    #Ejemplo
    st.markdown("#### Ejemplo de estimación para los siguientes datos:")
    sample_age = 24
    sample_children = 2
    sample_bmi = 30
    
    st.write({
        "Edad": sample_age,
        "Hijos": sample_children,
        "Indice de masa corporal": sample_bmi
    })
    
    st.markdown(f"## Costo estimado de poliza = ${(bmi*sample_bmi + children*sample_children + age*sample_age + model.intercept_)[0]:#.3f}")
    
    
    st.markdown("""---""")
    st.title("Prueba AB")
    
    st.write("""
             Lo que se busca con esta prueba es analizar que criterios proponen la mejor tasa
             de ventas de seguros, buscando el rendimiento entre ambas muestras.
             """)
        
    size = sms.proportion_effectsize(0.15, 0.065)
    required_size= sms.NormalIndPower().solve_power(
        effect_size=size, 
        power=0.8, 
        alpha=0.05,
        ratio=.5)
    
    size_required = ceil(required_size)
    st.write(f"Tamaño requerido:", size_required)
           
    #read the dataframes
    df = pd.read_csv("insurance.csv")
    #st.write(df[df["region"]=="southwest"].sample(20, random_state=22))
    control_sample = df[df["region"] == 'southwest'].sample(size_required, random_state=22)
    treatment_sample = df[df["region"] == 'southeast'].sample(size_required, random_state=22)
    
    #Concatenar los valores de cada grupo
    ab_test = pd.concat([control_sample, treatment_sample], axis=0)
  
    # calculate the better grade according the conversion grade
    conversion_grade = ab_test.groupby('region')['charges']
    
    # Standard Deviation function definition
    std_p = lambda x: np.std(x, ddof=0)
        
    #Deviation standard of the data errors
    e_p = lambda x:stats.sem(x, ddof=0)
    
    #comprobation of the Hypotesis
    control_results=ab_test[ab_test['region']=='southwest']['children']
    treatment_results=ab_test[ab_test['region']=='southeast']['children']
    
    n_control = control_results.count()
    n_tratamiento = treatment_results.count()
    exito = [control_results.sum(), treatment_results.sum()]
    nobs = [n_control,n_tratamiento]
    
    
    #Normal test
    from statsmodels.stats.proportion import proportions_ztest,proportion_confint
    z_stat, pval = proportions_ztest([199, 219], nobs=nobs)
    (lower_con,lower_treat),(upper_con,upper_treat) = proportion_confint(exito,nobs=nobs,alpha=0.05)

    st.write("z estadistico:", z_stat)
    st.write("p-value:", pval)
    st.write("Intervalo de confianza para control:",lower_con,upper_con)
    st.write("Intervalo de confianza para tratamiento:",lower_treat,upper_treat)
    
    st.markdown("### No hay evidencia suficiente para rechazar la hipotesis nula dado que p-value > 0.05")
    
    st.markdown("# Conclusion")
    
    st.write("""
             Las pruebas AB nos sirven para determinar y demostrar que 
             las suposiciones que tenemos son a menudo incorrectas.
             Una vez iniciada la prueba, el análisis ha de desempeñar un rol fundamental.
             Al final, es un approach cuantitativo que puede medir patrones de comportamiento de nuestros resultados y
             proveer los insights para desarrollar soluciones.
             """)