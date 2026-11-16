# Guía de selección de pruebas estadísticas

## Preguntas guía
Para elegir una prueba conviene responder cuatro preguntas. Primero, qué se quiere hacer: comparar grupos, medir asociación o estimar un parámetro. Segundo, qué escala tiene la variable de respuesta. Tercero, cuántos grupos hay y si son independientes o pareados. Cuarto, si se cumplen los supuestos de la prueba paramétrica.

## Tabla de selección
Comparar una media con un valor de referencia: t de una muestra; alternativa no paramétrica, Wilcoxon de una muestra.
Comparar dos grupos independientes: t de dos muestras (preferiblemente Welch); alternativa, U de Mann-Whitney.
Comparar dos mediciones pareadas: t pareada; alternativa, Wilcoxon de rangos con signo.
Comparar tres o más grupos independientes: ANOVA de un factor; alternativa, Kruskal-Wallis.
Medir la relación entre dos variables cuantitativas: correlación de Pearson; alternativa, Spearman.
Evaluar la asociación entre dos variables categóricas: chi-cuadrado de independencia; con frecuencias esperadas pequeñas, prueba exacta de Fisher.
Comparar una variable categórica con una distribución esperada: chi-cuadrado de bondad de ajuste.

## Ejemplos resueltos
Ejemplo 1. Se quiere comparar el tiempo de falla de componentes de tres proveedores. La variable de respuesta es de razón y hay tres grupos independientes: corresponde ANOVA de un factor si se cumplen los supuestos, y Kruskal-Wallis si no se cumplen.

Ejemplo 2. Se quiere comparar la nota de los mismos estudiantes antes y después de una tutoría. Son dos mediciones pareadas: corresponde la prueba t pareada, o Wilcoxon si no hay normalidad en las diferencias.

Ejemplo 3. Se quiere saber si el género (nominal) está asociado con aprobar o no una asignatura (sí o no). Son dos variables categóricas: corresponde la prueba chi-cuadrado de independencia.
