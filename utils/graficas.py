import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from metodos.interpolacion import evaluar_polinomio

def graficar_funcion(funcion_str, a, b, raiz=None):
    x = sp.symbols("x")
    f_sym = sp.sympify(funcion_str)
    f = sp.lambdify(x, f_sym, "numpy")

    xs = np.linspace(a, b, 400)
    ys = f(xs)

    fig, ax = plt.subplots()
    ax.plot(xs, ys, label="f(x)")
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    if raiz is not None:
        ax.scatter([raiz], [0], label="Raíz aproximada")

    ax.set_title("Gráfica de la función")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.legend()
    return fig

def graficar_errores(tabla):
    iteraciones = [fila["Iteración"] for fila in tabla]
    errores = [fila["Error"] for fila in tabla]

    fig, ax = plt.subplots()
    ax.plot(iteraciones, errores, marker="o")
    ax.set_title("Error por iteración")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Error")
    ax.grid(True)
    return fig

def graficar_interpolacion(xs, ys, polinomio, tipo="polinomio"):
    orden = np.argsort(xs)
    xs = xs[orden]
    ys = ys[orden]

    x_vals = np.linspace(min(xs), max(xs), 300)

    if tipo == "polinomio":
        y_vals = evaluar_polinomio(polinomio, x_vals)
    else:
        y_vals = polinomio(x_vals)

    fig, ax = plt.subplots()
    ax.scatter(xs, ys, label="Datos originales")
    ax.plot(x_vals, y_vals, label="Interpolación")
    ax.set_title("Interpolación")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend()
    return fig
