# Proyecto Final - Análisis Numérico

Aplicación desarrollada en Python con Streamlit para ejecutar métodos numéricos vistos durante el curso.

## Integrantes

- Daniel Urbano Viana Polo

## Capítulos incluidos

### Capítulo 1: Raíces de ecuaciones

Métodos implementados:

- Bisección
- Regla falsa
- Punto fijo
- Newton
- Secante
- Raíces múltiples

La interfaz permite ingresar funciones algebraicas, calcular derivadas automáticamente, visualizar la tabla de iteraciones, graficar la función y observar el error por iteración.

### Capítulo 2: Sistemas de ecuaciones lineales

Métodos implementados:

- Jacobi
- Gauss-Seidel
- SOR

La interfaz permite ingresar matrices hasta de tamaño 8x8, calcular el radio espectral, informar si el método puede converger y mostrar la tabla de iteraciones.

### Capítulo 3: Interpolación

Métodos implementados:

- Vandermonde
- Newton interpolante
- Lagrange
- Spline lineal
- Spline cúbico

La interfaz permite ingresar hasta 10 puntos, graficar la interpolación, mostrar el polinomio o función solución y comparar errores con porcentajes de datos almacenados.

## Instalación

Primero instala las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

Para ejecutar la aplicación:

```bash
streamlit run app.py
```