# Pruebas no paramétricas y para datos categóricos

## Cuándo usar pruebas no paramétricas
Las pruebas no paramétricas se usan cuando la variable es ordinal, cuando no se cumple la normalidad con muestras pequeñas o cuando hay valores atípicos fuertes. Requieren menos supuestos, pero tienen menor potencia que las pruebas paramétricas cuando los supuestos de estas sí se cumplen.

## Equivalencias entre pruebas
Cada prueba paramétrica tiene una alternativa no paramétrica. La prueba t para dos muestras independientes se reemplaza por la U de Mann-Whitney. La prueba t pareada se reemplaza por la prueba de rangos con signo de Wilcoxon. El ANOVA de un factor se reemplaza por la prueba de Kruskal-Wallis. La correlación de Pearson se reemplaza por la correlación de Spearman.

## Mann-Whitney y Wilcoxon
La U de Mann-Whitney compara dos grupos independientes usando los rangos de las observaciones, y evalúa si los valores de un grupo tienden a ser mayores que los del otro. La prueba de Wilcoxon de rangos con signo compara dos mediciones pareadas.

## Kruskal-Wallis
Compara tres o más grupos independientes usando rangos. Si rechaza la hipótesis nula, se usa una prueba post hoc como la de Dunn para identificar qué grupos difieren.

## Chi-cuadrado de independencia
Evalúa si dos variables categóricas están asociadas, a partir de una tabla de contingencia. La hipótesis nula es que las variables son independientes. El estadístico es la suma de (observado − esperado) al cuadrado dividido entre esperado. La frecuencia esperada de cada celda es el total de la fila por el total de la columna dividido entre n. Los grados de libertad son (filas − 1) por (columnas − 1): una tabla de 3 filas y 4 columnas tiene 6 grados de libertad.

El supuesto principal es que las frecuencias esperadas sean suficientemente grandes: se recomienda que la mayoría de las celdas tengan frecuencia esperada de al menos 5. Si no se cumple en una tabla de 2 por 2, se usa la prueba exacta de Fisher.

## Chi-cuadrado de bondad de ajuste
Compara las frecuencias observadas de una variable categórica con las frecuencias esperadas bajo una distribución teórica. Sus grados de libertad son el número de categorías menos 1, menos el número de parámetros estimados a partir de los datos.
