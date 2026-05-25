import numpy as np
import sympy as sp
from scipy.interpolate import interp1d, CubicSpline

x = sp.symbols("x")

def vandermonde(xs, ys):
    coef = np.polyfit(xs, ys, len(xs) - 1)
    polinomio = np.poly1d(coef)
    return polinomio

def newton_interpolante(xs, ys):
    n = len(xs)
    coef = np.copy(ys).astype(float)

    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (xs[j:n] - xs[0:n-j])

    polinomio = coef[0]
    producto = 1

    for i in range(1, n):
        producto *= (x - xs[i-1])
        polinomio += coef[i] * producto

    return sp.expand(polinomio)

def lagrange_interpolante(xs, ys):
    polinomio = 0

    for i in range(len(xs)):
        termino = ys[i]
        for j in range(len(xs)):
            if i != j:
                termino *= (x - xs[j]) / (xs[i] - xs[j])
        polinomio += termino

    return sp.expand(polinomio)

def spline_lineal(xs, ys):
    orden = np.argsort(xs)
    xs = xs[orden]
    ys = ys[orden]
    return interp1d(xs, ys, kind="linear", fill_value="extrapolate")

def spline_cubico(xs, ys):
    orden = np.argsort(xs)
    xs = xs[orden]
    ys = ys[orden]
    return CubicSpline(xs, ys)

def evaluar_polinomio(polinomio, valores_x):
    if isinstance(polinomio, np.poly1d):
        return polinomio(valores_x)

    if isinstance(polinomio, sp.Expr):
        f = sp.lambdify(x, polinomio, "numpy")
        return f(valores_x)

    return polinomio(valores_x)
