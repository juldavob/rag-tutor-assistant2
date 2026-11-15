# Correlación y regresión lineal simple

## Correlación de Pearson
El coeficiente de correlación de Pearson, r, mide la fuerza de la asociación lineal entre dos variables cuantitativas y toma valores entre −1 y 1. Supone una relación lineal y distribución aproximadamente normal conjunta, y es sensible a valores atípicos. Un valor de r igual a 0 no implica independencia: las variables pueden tener una relación no lineal fuerte, como una parábola.

## Correlación de Spearman
La correlación de Spearman se calcula sobre los rangos de los datos. Se usa con variables ordinales o cuando la relación es monótona pero no lineal, y es más robusta frente a atípicos que la de Pearson.

## Correlación y causalidad
Una correlación alta no demuestra que una variable cause la otra: pueden existir variables de confusión o una relación inversa.

## Regresión lineal simple
El modelo es y = β0 + β1·x + ε. Los coeficientes se estiman por mínimos cuadrados. La pendiente β1 indica el cambio promedio en y por cada unidad adicional de x. El coeficiente de determinación R cuadrado es la proporción de la variabilidad de y explicada por el modelo; en regresión lineal simple, R cuadrado es igual al cuadrado de r.

## Inferencia sobre la pendiente
Para probar si existe relación lineal se contrasta H0: β1 = 0 con una prueba t. También se puede construir un intervalo de confianza para β1: si no contiene el cero, hay evidencia de relación lineal.

## Supuestos del modelo
Los supuestos son linealidad de la relación, independencia de los errores, varianza constante de los errores (homocedasticidad) y normalidad de los residuos. Se revisan con gráficos de residuos.

## Extrapolación
Predecir para valores de x fuera del rango observado es riesgoso, porque no hay evidencia de que la relación lineal se mantenga allí.
