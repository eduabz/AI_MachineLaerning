# Predicción de volumen de tráfico

## Descripción

Este proyecto busca predecir el volumen de tráfico por hora en la carretera I-94 usando información relacionada con el clima, la hora y el día.

Para esta entrega se implementó manualmente una regresión lineal múltiple sin utilizar frameworks.

## Dataset

Se utilizó el dataset Metro Interstate Traffic Volume.

Las variables usadas para el modelo fueron:

- holiday
- temp
- rain_1h
- clouds_all
- hour
- weekday

La variable que se busca predecir es:

- traffic_volume

## Limpieza de datos

Antes de entrenar el modelo se realizaron algunas transformaciones básicas:

- Se convirtió `holiday` a valores 0 y 1.
- Se eliminaron registros con temperatura igual a 0.
- Se eliminó la variable `snow_1h`.
- Se eliminó `weather_description`.
- Se eliminó un valor anómalo de lluvia de 9831.3.
- Se dejó una sola observación por hora.
- Se extrajeron la hora y el día de la semana desde `date_time`.
- Las variables numéricas usadas en el modelo fueron normalizadas entre 0 y 1.

## Algoritmo

Se implementó una regresión lineal múltiple usando descenso de gradiente.

La predicción se calcula con:

y_pred = w1*x1 + w2*x2 + ... + wn*xn + b

Los pesos y el bias empiezan en 0 y se van actualizando durante el entrenamiento con base en el error de las predicciones.

Para medir el error se utilizó MSE (Mean Squared Error).

## Resultado

Después del entrenamiento se obtuvo un MSE aproximado de:

3646772.65

Se simularon 10 predicciones y los resultados fueron los siguientes: 

- Real: 5545 - Predicción: 3010.62
- Real: 4516 - Predicción: 3307.53
- Real: 4767 - Predicción: 3457.8
- Real: 5026 - Predicción: 3512.1
- Real: 4918 - Predicción: 3472.32
- Real: 5181 - Predicción: 3030.24
- Real: 5584 - Predicción: 3097.48
- Real: 6015 - Predicción: 3153.8
- Real: 5791 - Predicción: 3331.79
- Real: 4770 - Predicción: 3363.24

El modelo logra realizar predicciones, aunque todavía presenta un error considerable.
## Instalación

Instalar las librerías necesarias:

```bash
pip install pandas numpy
```

## Ejecución

```bash
python3 traffic_modelo.py