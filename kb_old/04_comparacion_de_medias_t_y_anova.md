# Comparación de medias: pruebas t y ANOVA

## Prueba t para una muestra
Compara la media de una muestra con un valor de referencia μ0. El estadístico es t = (media muestral − μ0) / (s / raíz de n), con n − 1 grados de libertad. Supone observaciones independientes y distribución aproximadamente normal, o una muestra grande.

## Prueba t para dos muestras independientes
Compara las medias de dos grupos distintos, por ejemplo dos métodos de fabricación. La versión de Student supone varianzas iguales en los dos grupos; la versión de Welch no lo supone y suele recomendarse por defecto. Los supuestos son independencia entre las observaciones y normalidad aproximada en cada grupo. La igualdad de varianzas se puede verificar con la prueba de Levene.

## Prueba t pareada
Se usa cuando se tienen mediciones repetidas sobre las mismas unidades, por ejemplo antes y después de un tratamiento. Se calcula la diferencia dentro de cada pareja y se aplica una prueba t de una muestra a esas diferencias con μ0 = 0. El supuesto de normalidad se aplica a las diferencias.

## ANOVA de un factor
El análisis de varianza de un factor compara las medias de tres o más grupos independientes. La hipótesis nula es que todas las medias son iguales y la alternativa es que al menos una media es distinta. El estadístico es F, el cociente entre el cuadrado medio entre grupos y el cuadrado medio dentro de los grupos. Los supuestos son independencia, normalidad de los residuos y homogeneidad de varianzas.

## Por qué no repetir pruebas t
Comparar tres grupos mediante varias pruebas t infla el error tipo I global. Con tres grupos hay tres comparaciones por pares; si cada una usa α = 0,05, la probabilidad de cometer al menos un falso positivo es aproximadamente 1 − 0,95 al cubo, es decir, cerca de 0,14. ANOVA evita este problema porque realiza una sola prueba global.

## Pruebas post hoc
Cuando ANOVA rechaza la hipótesis nula, solo indica que al menos una media difiere, pero no cuál. Para identificar qué pares de grupos difieren se usan pruebas post hoc como Tukey HSD, que controlan el error global de las comparaciones por pares.

## Verificación de supuestos
La normalidad se revisa con la prueba de Shapiro-Wilk y con gráficos Q-Q. La homogeneidad de varianzas se revisa con la prueba de Levene. Si los supuestos no se cumplen, se usan las alternativas no paramétricas.
