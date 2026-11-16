# Estimación e intervalos de confianza

## Estimación puntual
Un estimador es un estadístico calculado a partir de la muestra para aproximar un parámetro de la población. La media muestral estima la media poblacional, la proporción muestral estima la proporción poblacional y la varianza muestral estima la varianza poblacional.

La varianza muestral usa n − 1 en el divisor en lugar de n. Con este divisor la varianza muestral es un estimador insesgado de la varianza poblacional, es decir, su valor esperado es igual al parámetro. Dividir entre n produciría una subestimación sistemática.

## Error estándar
El error estándar es la desviación estándar de la distribución muestral de un estimador. Para la media muestral se calcula como s dividido entre la raíz de n. Disminuye cuando aumenta el tamaño de muestra, pero lo hace con la raíz de n: para reducir el error estándar a la mitad hay que multiplicar el tamaño de muestra por cuatro.

## Intervalo de confianza para la media
Cuando la desviación estándar poblacional es desconocida, que es el caso habitual, el intervalo de confianza para la media es la media muestral más o menos t multiplicado por s dividido entre la raíz de n, donde t es el valor crítico de la distribución t de Student con n − 1 grados de libertad. Este intervalo supone que la población es aproximadamente normal o que la muestra es grande. Si la desviación estándar poblacional fuera conocida se usaría el valor crítico z de la normal.

## Interpretación del intervalo de confianza
Un intervalo de confianza del 95% significa que, si se repitiera el muestreo muchas veces y se construyera un intervalo en cada ocasión, aproximadamente el 95% de esos intervalos contendría el valor verdadero del parámetro. No significa que exista un 95% de probabilidad de que el parámetro esté dentro de un intervalo ya calculado: el parámetro es un valor fijo y lo aleatorio es el intervalo.

## Factores que afectan el ancho del intervalo
El intervalo se ensancha al aumentar el nivel de confianza y al aumentar la variabilidad de los datos. El intervalo se estrecha al aumentar el tamaño de muestra.

## Tamaño de muestra para estimar una media
Para estimar una media con un error máximo E y un nivel de confianza dado, el tamaño de muestra es n = (z · σ / E) al cuadrado, redondeado siempre hacia arriba. Ejemplo: con σ = 10, E = 2 y 95% de confianza (z = 1,96), n = (1,96 × 10 / 2) al cuadrado = 96,04, por lo que se necesitan 97 observaciones.

## Intervalo de confianza para una proporción
El intervalo para una proporción es la proporción muestral más o menos z multiplicado por la raíz de p-sombrero por (1 − p-sombrero) dividido entre n. Se aplica cuando n por p-sombrero y n por (1 − p-sombrero) son ambos al menos 10.
