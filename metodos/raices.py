import sympy as sp
import numpy as np

x = sp.symbols("x")

def _funcion(funcion_str):
    f_sym = sp.sympify(funcion_str)
    return sp.lambdify(x, f_sym, "numpy"), f_sym

def _error(x_actual, x_anterior, tipo):
    if x_anterior is None:
        return None
    if tipo == "relativo":
        if x_actual == 0:
            return abs(x_actual - x_anterior)
        return abs((x_actual - x_anterior) / x_actual)
    return abs(x_actual - x_anterior)

def biseccion(funcion_str, a, b, tol, max_iter, tipo_error="absoluto"):
    f, _ = _funcion(funcion_str)

    if f(a) * f(b) > 0:
        raise ValueError("El intervalo no cumple f(a)*f(b) < 0.")

    tabla = []
    xm_anterior = None

    for i in range(1, max_iter + 1):
        xm = (a + b) / 2
        fxm = f(xm)
        error = _error(xm, xm_anterior, tipo_error)

        tabla.append({
            "Iteración": i,
            "a": a,
            "b": b,
            "xm": xm,
            "f(xm)": fxm,
            "Error": error
        })

        if abs(fxm) < tol or (error is not None and error < tol):
            return xm, tabla

        if f(a) * fxm < 0:
            b = xm
        else:
            a = xm

        xm_anterior = xm

    return xm, tabla

def regla_falsa(funcion_str, a, b, tol, max_iter, tipo_error="absoluto"):
    f, _ = _funcion(funcion_str)

    if f(a) * f(b) > 0:
        raise ValueError("El intervalo no cumple f(a)*f(b) < 0.")

    tabla = []
    xr_anterior = None

    for i in range(1, max_iter + 1):
        xr = b - (f(b) * (a - b)) / (f(a) - f(b))
        fxr = f(xr)
        error = _error(xr, xr_anterior, tipo_error)

        tabla.append({
            "Iteración": i,
            "a": a,
            "b": b,
            "xr": xr,
            "f(xr)": fxr,
            "Error": error
        })

        if abs(fxr) < tol or (error is not None and error < tol):
            return xr, tabla

        if f(a) * fxr < 0:
            b = xr
        else:
            a = xr

        xr_anterior = xr

    return xr, tabla

def punto_fijo(g_str, x0, tol, max_iter, tipo_error="absoluto"):
    g, _ = _funcion(g_str)
    tabla = []
    x_anterior = x0

    for i in range(1, max_iter + 1):
        x_actual = g(x_anterior)
        error = _error(x_actual, x_anterior, tipo_error)

        tabla.append({
            "Iteración": i,
            "x anterior": x_anterior,
            "x actual": x_actual,
            "Error": error
        })

        if error < tol:
            return x_actual, tabla

        x_anterior = x_actual

    return x_actual, tabla

def newton(funcion_str, x0, tol, max_iter, tipo_error="absoluto"):
    f, f_sym = _funcion(funcion_str)
    df_sym = sp.diff(f_sym, x)
    df = sp.lambdify(x, df_sym, "numpy")

    tabla = []
    x_anterior = x0

    for i in range(1, max_iter + 1):
        if df(x_anterior) == 0:
            raise ValueError("La derivada es cero. No se puede continuar.")

        x_actual = x_anterior - f(x_anterior) / df(x_anterior)
        error = _error(x_actual, x_anterior, tipo_error)

        tabla.append({
            "Iteración": i,
            "x anterior": x_anterior,
            "x actual": x_actual,
            "f(x)": f(x_actual),
            "Error": error
        })

        if error < tol:
            return x_actual, tabla

        x_anterior = x_actual

    return x_actual, tabla

def secante(funcion_str, x0, x1, tol, max_iter, tipo_error="absoluto"):
    f, _ = _funcion(funcion_str)
    tabla = []

    for i in range(1, max_iter + 1):
        if f(x1) - f(x0) == 0:
            raise ValueError("División por cero en el método de secante.")

        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        error = _error(x2, x1, tipo_error)

        tabla.append({
            "Iteración": i,
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "f(x2)": f(x2),
            "Error": error
        })

        if error < tol:
            return x2, tabla

        x0, x1 = x1, x2

    return x2, tabla

def raices_multiples(funcion_str, x0, tol, max_iter, tipo_error="absoluto"):
    f, f_sym = _funcion(funcion_str)
    df_sym = sp.diff(f_sym, x)
    ddf_sym = sp.diff(df_sym, x)

    df = sp.lambdify(x, df_sym, "numpy")
    ddf = sp.lambdify(x, ddf_sym, "numpy")

    tabla = []
    x_anterior = x0

    for i in range(1, max_iter + 1):
        denominador = df(x_anterior)**2 - f(x_anterior) * ddf(x_anterior)

        if denominador == 0:
            raise ValueError("El denominador es cero. No se puede continuar.")

        x_actual = x_anterior - (f(x_anterior) * df(x_anterior)) / denominador
        error = _error(x_actual, x_anterior, tipo_error)

        tabla.append({
            "Iteración": i,
            "x anterior": x_anterior,
            "x actual": x_actual,
            "f(x)": f(x_actual),
            "Error": error
        })

        if error < tol:
            return x_actual, tabla

        x_anterior = x_actual

    return x_actual, tabla
