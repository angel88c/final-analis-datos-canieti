
# Se desea confirmar si la nueva pagina va a atraer el 15%
# mas a los usuarios. 

# Inferencia estadística.

# Hipótesis

# Tablas de probabilidad

#Po = nueva pagina web 
#P = vieja pagina web
# Ho=P=Po - Hipotesis nula
# Hl=P!=Po - Hipotesis alternativa

###Librerias
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
from math import ceil



####Calcular e número de datos que necesito para que la prueba sea real
tamano=sms.proportion_effectsize(0.13,0.15)
tamano_requerido=sms.NormalIndPower().solve_power(
    tamano,
    power=0.8,
    alpha=0.05,
    ratio=1)

tamano_requerido=ceil(tamano_requerido)
print(tamano_requerido)

###Importar bases de datos
df=pd.read_csv("ab_data.csv")

##Verificar que los usiarios (user_id no estén duplicados para evitar errores)
sesiones=df['user_id'].value_counts(ascending=False)
sesiones

####Sacar los usuarios repetidos 

usuarios_repetidos=sesiones[sesiones>1].count

###Eliminar los repetidos
usuarios_eliminar=sesiones[sesiones>1].index
df=df[~df['user_id'].isin(usuarios_eliminar)]


###Obtener los 4720 usuarios de cada grupo
muestra_control=df[df['group']=='control'].sample(tamano_requerido,random_state=22)
muestra_tratamiento=df[df['group']=='treatment'].sample(tamano_requerido,random_state=22)


#Concatenar valores de cada grupo
ab_prueba=pd.concat([muestra_control,muestra_tratamiento],axis=0)




### Calcular el grupo ques es mejor de acuerdo a su grado de conversion
grado_conversion=ab_prueba.groupby('group')['converted']



#### Vamos a calcular el nivel de la desviación estandar de los datos
std_p=lambda x:np.std(x,ddof=0)
print(std_p)




##desviacion estandar de los errores de los datos
se_p=lambda x:stats.sem(x,ddof=0)




#### grado de conversion final

grado_conversion=grado_conversion.agg([np.mean,std_p,se_p])





#comprobacion de la hipotesis p-value

resultados_control=ab_prueba[ab_prueba['group']=='control']['converted']
resultados_tratamiento=ab_prueba[ab_prueba['group']=='treatment']['converted']
n_control=resultados_control.count()
n_tratamiento=resultados_tratamiento.count()
exito=[resultados_control.sum(),resultados_tratamiento.sum()]
nobs=[n_control,n_tratamiento]


print(exito)
print(nobs)


##### normal test

from statsmodels.stats.proportion import proportions_ztest,proportion_confint
z_stat,pval=proportions_ztest(exito,nobs=nobs)
(lower_con,lower_treat),(upper_con,upper_treat)=proportion_confint(exito,nobs=nobs,alpha=0.05)

##### los valores de la prueba


print("z estadistico:",z_stat)
print("p-value:",pval)
print("intervalo de confianza para control:",lower_con,upper_con)
print("Intervalo de confianza para tratamiento:",lower_treat,upper_treat)


###### Ho se rechace, para que con eso se cumpla que en un 15% se uso de mas 



#####  p-value < alpha (0.05)------> se rechaza la hipotesis nula
#####   p-value > alpha (0.05) ------> no hay sufieciente evidencia estadistica para rechazar la hipotesis nula 

####### 0.73>0.05-------------------> no hay suficiente evidencia estadistica para rechazar la hipotesis nula



######### Ho=P=Po    (Hipotesis nula Ho)  rechazar la Ho
######### H1=P≠Po (Hipotesis alternativa)