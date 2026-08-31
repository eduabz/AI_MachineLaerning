import pandas as pd
import numpy as np

df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

# ETL del dataset 

df["holiday"] = df["holiday"].fillna(0)
df.loc[df["holiday"] != 0, "holiday"] = 1
df["holiday"] = df["holiday"].astype(int)

df = df[df["temp"] != 0]

df = df.drop(columns=["snow_1h"])

df = df.drop(columns=["weather_description"])

df = df[df["rain_1h"] != 9831.3]

df = df.drop_duplicates(subset="date_time", keep="first")

df["date_time"] = pd.to_datetime(df["date_time"])

df["hour"] = df["date_time"].dt.hour
df["weekday"] = df["date_time"].dt.dayofweek

# Preparamos las variables de entrada y objetivo 

X = df[
    [
        "holiday",
        "temp",
        "rain_1h",
        "clouds_all",
        "hour",
        "weekday"
    ]
]

y = df["traffic_volume"]

#Normalizacion de las variables 

columnas_normalizar = [
    "temp",
    "rain_1h",
    "clouds_all",
    "hour",
    "weekday"
]

X_normalizado = X.copy()

for columna in columnas_normalizar:

    minimo = X_normalizado[columna].min()
    maximo = X_normalizado[columna].max()

    X_normalizado[columna] = (
        X_normalizado[columna] - minimo
    ) / (maximo - minimo)

    datos = X_normalizado.values
objetivos = y.values

print("Datos preparados correctamente")
print("Cantidad de registros:", len(datos))

# Inicializamos pesos y bias 

pesos = np.zeros(X_normalizado.shape[1])
b = 0

learning_rate = 0.01
epocas = 100

# Descenso de gradiente manual 

for epoca in range(epocas):

    predicciones = np.dot(datos, pesos) + b

    errores = predicciones - objetivos

    gradiente_pesos = (2 / len(datos)) * np.dot(datos.T, errores)
    gradiente_b = (2 / len(datos)) * np.sum(errores)

    pesos = pesos - learning_rate * gradiente_pesos
    b = b - learning_rate * gradiente_b

print("\nEntrenamiento terminado")

print("Pesos:")
print(pesos)

print("Bias:")
print(b)    

# Preedicciones 

predicciones = np.dot(datos, pesos) + b


# MSE 

errores = predicciones - objetivos

mse = np.mean(errores ** 2)

print("\nMSE( Mean Squared Error):")
print(mse)


print("\nSe simularon 10 predicciones y los resultados fueron los siguientes: :")

for i in range(10):
    print(
        "Real:", objetivos[i],
        "- Predicción:", round(predicciones[i], 2)
    )