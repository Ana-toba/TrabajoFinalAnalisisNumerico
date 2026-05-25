import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt

from metodos.raices import (
    biseccion, regla_falsa, punto_fijo, newton, secante, raices_multiples
)
from metodos.sistemas import jacobi, gauss_seidel, sor, radio_espectral_iteracion
from metodos.interpolacion import (
    vandermonde, newton_interpolante, lagrange_interpolante,
    spline_lineal, spline_cubico, evaluar_polinomio
)
from utils.graficas import graficar_funcion, graficar_errores, graficar_interpolacion

st.set_page_config(page_title="Análisis Numérico", layout="wide")

st.title("Proyecto Final - Análisis Numérico")
st.write("Aplicación para ejecutar métodos numéricos vistos durante el curso.")

st.sidebar.header("Menú")
capitulo = st.sidebar.radio(
    "Selecciona un capítulo",
    ["Capítulo 1: Raíces", "Capítulo 2: Sistemas lineales", "Capítulo 3: Interpolación"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Integrante")
st.sidebar.write("• Daniel Urbano Viana Polo")


# ============================================================
# CAPÍTULO 1
# ============================================================
if capitulo == "Capítulo 1: Raíces":
    st.header("Capítulo 1: Métodos para encontrar raíces")

    with st.expander("Ayuda para ingresar funciones"):
        st.write("""
        Escribe la función usando sintaxis de Python/SymPy.

        Ejemplos:
        - `x**3 - x - 2`
        - `sin(x) - x/2`
        - `exp(x) - 3*x`
        - `log(x) - 1`

        Para punto fijo debes ingresar una función g(x), por ejemplo:
        - si tienes `x**3 - x - 2 = 0`
        - puedes usar `g(x) = (x + 2)**(1/3)`
        """)

    funcion_str = st.text_input("Función f(x)", "x**3 - x - 2")

    metodo = st.selectbox(
        "Método",
        ["Bisección", "Regla falsa", "Punto fijo", "Newton", "Secante", "Raíces múltiples"]
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        tol = st.number_input("Tolerancia", value=0.0001, format="%.8f")
    with col2:
        max_iter = st.number_input("Máximo de iteraciones", min_value=1, value=100)
    with col3:
        tipo_error = st.selectbox("Tipo de error", ["absoluto", "relativo"])

    try:
        x = sp.symbols("x")
        f_sym = sp.sympify(funcion_str)
        df_sym = sp.diff(f_sym, x)

        st.write("Derivada calculada automáticamente:")
        st.latex(f"f'(x) = {sp.latex(df_sym)}")

        if metodo in ["Bisección", "Regla falsa"]:
            col1, col2 = st.columns(2)
            with col1:
                a = st.number_input("Valor inferior a", value=1.0)
            with col2:
                b = st.number_input("Valor superior b", value=2.0)

            if st.button("Ejecutar método"):
                if metodo == "Bisección":
                    resultado, tabla = biseccion(funcion_str, a, b, tol, max_iter, tipo_error)
                else:
                    resultado, tabla = regla_falsa(funcion_str, a, b, tol, max_iter, tipo_error)

                st.success(f"Raíz aproximada: {resultado}")
                st.dataframe(pd.DataFrame(tabla))
                st.pyplot(graficar_funcion(funcion_str, a, b, resultado))
                st.pyplot(graficar_errores(tabla))

        elif metodo == "Punto fijo":
            st.info("Para punto fijo debes ingresar una función g(x), donde x = g(x).")
            g_str = st.text_input("Función g(x)", "(x + 2)**(1/3)")
            x0 = st.number_input("Valor inicial x0", value=1.0)

            if st.button("Ejecutar método"):
                resultado, tabla = punto_fijo(g_str, x0, tol, max_iter, tipo_error)
                st.success(f"Raíz aproximada: {resultado}")
                st.dataframe(pd.DataFrame(tabla))
                st.pyplot(graficar_errores(tabla))

        elif metodo in ["Newton", "Raíces múltiples"]:
            x0 = st.number_input("Valor inicial x0", value=1.0)

            if st.button("Ejecutar método"):
                if metodo == "Newton":
                    resultado, tabla = newton(funcion_str, x0, tol, max_iter, tipo_error)
                else:
                    resultado, tabla = raices_multiples(funcion_str, x0, tol, max_iter, tipo_error)

                st.success(f"Raíz aproximada: {resultado}")
                st.dataframe(pd.DataFrame(tabla))
                st.pyplot(graficar_funcion(funcion_str, x0 - 5, x0 + 5, resultado))
                st.pyplot(graficar_errores(tabla))

        elif metodo == "Secante":
            col1, col2 = st.columns(2)
            with col1:
                x0 = st.number_input("Valor inicial x0", value=1.0)
            with col2:
                x1 = st.number_input("Valor inicial x1", value=2.0)

            if st.button("Ejecutar método"):
                resultado, tabla = secante(funcion_str, x0, x1, tol, max_iter, tipo_error)
                st.success(f"Raíz aproximada: {resultado}")
                st.dataframe(pd.DataFrame(tabla))
                st.pyplot(graficar_funcion(funcion_str, min(x0, x1) - 5, max(x0, x1) + 5, resultado))
                st.pyplot(graficar_errores(tabla))

        st.markdown("---")
        st.subheader("Comparación de errores entre métodos")

        with st.expander("Comparar todos los métodos de raíces"):
            st.write("Esta sección ejecuta varios métodos con los mismos parámetros y compara sus errores por iteración.")

            col1, col2, col3 = st.columns(3)
            with col1:
                comp_a = st.number_input("a para comparación", value=1.0)
            with col2:
                comp_b = st.number_input("b para comparación", value=2.0)
            with col3:
                comp_x0 = st.number_input("x0 para comparación", value=1.0)

            comp_x1 = st.number_input("x1 para secante", value=2.0)
            comp_g = st.text_input("g(x) para punto fijo", "(x + 2)**(1/3)")

            errores_comparar = st.multiselect(
                "Tipos de error a comparar",
                ["absoluto", "relativo"],
                default=["absoluto", "relativo"]
            )

            if st.button("Comparar métodos de raíces"):
                for error_tipo in errores_comparar:
                    resultados = {}

                    try:
                        _, tabla_bis = biseccion(funcion_str, comp_a, comp_b, tol, max_iter, error_tipo)
                        resultados["Bisección"] = tabla_bis
                    except Exception:
                        pass

                    try:
                        _, tabla_rf = regla_falsa(funcion_str, comp_a, comp_b, tol, max_iter, error_tipo)
                        resultados["Regla falsa"] = tabla_rf
                    except Exception:
                        pass

                    try:
                        _, tabla_pf = punto_fijo(comp_g, comp_x0, tol, max_iter, error_tipo)
                        resultados["Punto fijo"] = tabla_pf
                    except Exception:
                        pass

                    try:
                        _, tabla_newton = newton(funcion_str, comp_x0, tol, max_iter, error_tipo)
                        resultados["Newton"] = tabla_newton
                    except Exception:
                        pass

                    try:
                        _, tabla_sec = secante(funcion_str, comp_a, comp_x1, tol, max_iter, error_tipo)
                        resultados["Secante"] = tabla_sec
                    except Exception:
                        pass

                    try:
                        _, tabla_rm = raices_multiples(funcion_str, comp_x0, tol, max_iter, error_tipo)
                        resultados["Raíces múltiples"] = tabla_rm
                    except Exception:
                        pass

                    if resultados:
                        fig_comp, ax = plt.subplots()

                        for nombre, tabla in resultados.items():
                            iteraciones = [fila["Iteración"] for fila in tabla]
                            errores = [
                                fila["Error"] if fila["Error"] is not None else np.nan
                                for fila in tabla
                            ]
                            ax.plot(iteraciones, errores, marker="o", label=nombre)

                        ax.set_title(f"Comparación de errores - Error {error_tipo}")
                        ax.set_xlabel("Iteración")
                        ax.set_ylabel("Error")
                        ax.grid(True)
                        ax.legend()
                        st.pyplot(fig_comp)
                    else:
                        st.warning("No fue posible ejecutar los métodos con los parámetros ingresados.")

    except Exception as e:
        st.error(f"Error: {e}")


# ============================================================
# CAPÍTULO 2
# ============================================================
elif capitulo == "Capítulo 2: Sistemas lineales":
    st.header("Capítulo 2: Métodos iterativos para sistemas lineales")

    with st.expander("Ayuda para ingresar matrices"):
        st.write("""
        Ingresa la matriz A y el vector b separados por comas.

        Ejemplo:

        A:
        ```
        4, -1, 0
        -1, 4, -1
        0, -1, 4
        ```

        b:
        ```
        15, 10, 10
        ```

        El sistema se interpreta como A·x = b.
        La matriz debe ser cuadrada y puede tener tamaño máximo 8x8.
        """)

    n = st.number_input("Tamaño de la matriz", min_value=2, max_value=8, value=3)

    matriz_default = "4, -1, 0\n-1, 4, -1\n0, -1, 4"
    vector_default = "15, 10, 10"

    A_text = st.text_area("Matriz A", matriz_default)
    b_text = st.text_area("Vector b", vector_default)

    metodo = st.selectbox("Método", ["Jacobi", "Gauss-Seidel", "SOR"])

    col1, col2, col3 = st.columns(3)
    with col1:
        tol = st.number_input("Tolerancia", value=0.0001, format="%.8f")
    with col2:
        max_iter = st.number_input("Máximo de iteraciones", min_value=1, value=100)
    with col3:
        tipo_error = st.selectbox("Tipo de error", ["absoluto", "relativo", "condicion"])

    w = 1.0
    if metodo == "SOR":
        w = st.number_input("Parámetro de relajación w", min_value=0.1, max_value=2.0, value=1.1)

    if st.button("Ejecutar sistema"):
        try:
            A = np.array([[float(num) for num in fila.split(",")] for fila in A_text.strip().split("\n")])
            b = np.array([float(num) for num in b_text.replace("\n", "").split(",")])
            x0 = np.zeros(len(b))

            if A.shape[0] != A.shape[1]:
                st.error("La matriz A debe ser cuadrada.")
            elif A.shape[0] != len(b):
                st.error("El tamaño de A y b no coincide.")
            elif A.shape[0] > 8:
                st.error("La matriz no puede superar tamaño 8x8.")
            else:
                if metodo == "Jacobi":
                    resultado, tabla = jacobi(A, b, x0, tol, max_iter, tipo_error)
                    rho = radio_espectral_iteracion(A, "jacobi")
                elif metodo == "Gauss-Seidel":
                    resultado, tabla = gauss_seidel(A, b, x0, tol, max_iter, tipo_error)
                    rho = radio_espectral_iteracion(A, "gauss-seidel")
                else:
                    resultado, tabla = sor(A, b, x0, w, tol, max_iter, tipo_error)
                    rho = radio_espectral_iteracion(A, "sor", w)

                st.success(f"Solución aproximada: {resultado}")
                st.write(f"Radio espectral: {rho}")

                if rho < 1:
                    st.success("Según el radio espectral, el método puede converger.")
                else:
                    st.warning("Según el radio espectral, el método puede no converger.")

                st.dataframe(pd.DataFrame(tabla))
                st.pyplot(graficar_errores(tabla))

        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.subheader("Comparación de errores entre métodos")

    with st.expander("Comparar Jacobi, Gauss-Seidel y SOR"):
        st.write("Esta sección ejecuta los tres métodos con la misma matriz y compara sus errores por iteración.")

        errores_comparar_sistemas = st.multiselect(
            "Tipos de error a comparar",
            ["absoluto", "relativo", "condicion"],
            default=["absoluto", "relativo", "condicion"]
        )

        w_comp = st.number_input(
            "Parámetro w para comparación SOR",
            min_value=0.1,
            max_value=2.0,
            value=1.1,
            key="w_comparacion_sor"
        )

        if st.button("Comparar métodos de sistemas"):
            try:
                A = np.array([[float(num) for num in fila.split(",")] for fila in A_text.strip().split("\n")])
                b = np.array([float(num) for num in b_text.replace("\n", "").split(",")])
                x0 = np.zeros(len(b))

                if A.shape[0] != A.shape[1]:
                    st.error("La matriz A debe ser cuadrada.")
                elif A.shape[0] != len(b):
                    st.error("El tamaño de A y b no coincide.")
                elif A.shape[0] > 8:
                    st.error("La matriz no puede superar tamaño 8x8.")
                else:
                    for error_tipo in errores_comparar_sistemas:
                        resultados = {}

                        try:
                            _, tabla_jacobi = jacobi(A, b, x0, tol, max_iter, error_tipo)
                            resultados["Jacobi"] = tabla_jacobi
                        except Exception:
                            pass

                        try:
                            _, tabla_gs = gauss_seidel(A, b, x0, tol, max_iter, error_tipo)
                            resultados["Gauss-Seidel"] = tabla_gs
                        except Exception:
                            pass

                        try:
                            _, tabla_sor = sor(A, b, x0, w_comp, tol, max_iter, error_tipo)
                            resultados["SOR"] = tabla_sor
                        except Exception:
                            pass

                        if resultados:
                            fig_comp, ax = plt.subplots()

                            for nombre, tabla in resultados.items():
                                iteraciones = [fila["Iteración"] for fila in tabla]
                                errores = [fila["Error"] for fila in tabla]
                                ax.plot(iteraciones, errores, marker="o", label=nombre)

                            ax.set_title(f"Comparación de errores - Error {error_tipo}")
                            ax.set_xlabel("Iteración")
                            ax.set_ylabel("Error")
                            ax.grid(True)
                            ax.legend()

                            st.pyplot(fig_comp)
                        else:
                            st.warning(f"No fue posible comparar los métodos para error {error_tipo}.")

            except Exception as e:
                st.error(f"Error en la comparación: {e}")
# ============================================================
# CAPÍTULO 3
# ============================================================
elif capitulo == "Capítulo 3: Interpolación":
    st.header("Capítulo 3: Métodos de interpolación")

    with st.expander("Ayuda para ingresar datos"):
        st.write("""
        Ingresa hasta 10 puntos. Los valores de X y Y deben tener la misma cantidad de datos.

        Ejemplo:
        - X: `0,1,2,3,4,5,6,7,8,9`
        - Y: `1,3,2,5,4,6,7,8,7,10`

        Importante:
        - No repitas valores de X.
        - Para spline cúbico se recomienda ingresar mínimo 4 puntos.
        """)

    x_text = st.text_input("Valores de X", "0,1,2,3,4,5,6,7,8,9")
    y_text = st.text_input("Valores de Y", "1,3,2,5,4,6,7,8,7,10")

    metodo = st.selectbox(
        "Método",
        ["Vandermonde", "Newton interpolante", "Lagrange", "Spline lineal", "Spline cúbico"]
    )

    def texto_spline_lineal(xs, ys):
        tramos = []
        for i in range(len(xs) - 1):
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[i], ys[i + 1]
            m = (y1 - y0) / (x1 - x0)
            b = y0 - m * x0
            tramos.append(f"Para {x0} ≤ x ≤ {x1}:   S{i}(x) = {m:.6f}x + {b:.6f}")
        return "\n".join(tramos)

    def texto_spline_cubico(modelo, xs):
        tramos = []
        coef = modelo.c

        for i in range(len(xs) - 1):
            a = coef[0, i]
            b = coef[1, i]
            c = coef[2, i]
            d = coef[3, i]
            xi = xs[i]
            xf = xs[i + 1]

            tramos.append(
                f"Para {xi} ≤ x ≤ {xf}:   "
                f"S{i}(x) = {a:.6f}(x - {xi:.6f})³ + "
                f"{b:.6f}(x - {xi:.6f})² + "
                f"{c:.6f}(x - {xi:.6f}) + {d:.6f}"
            )

        return "\n".join(tramos)

    if st.button("Interpolar"):
        try:
            xs = np.array([float(i.strip()) for i in x_text.split(",")])
            ys = np.array([float(i.strip()) for i in y_text.split(",")])

            if len(xs) != len(ys):
                st.error("X y Y deben tener la misma cantidad de datos.")
            elif len(xs) > 10:
                st.error("Solo se permiten hasta 10 datos.")
            elif len(xs) < 2:
                st.error("Debes ingresar al menos 2 puntos.")
            elif len(np.unique(xs)) != len(xs):
                st.error("No puedes repetir valores de X.")
            elif metodo == "Spline cúbico" and len(xs) < 4:
                st.error("Para spline cúbico se recomienda ingresar mínimo 4 puntos.")
            else:
                orden = np.argsort(xs)
                xs = xs[orden]
                ys = ys[orden]

                if metodo == "Vandermonde":
                    modelo = vandermonde(xs, ys)
                    tipo = "polinomio"
                    solucion_texto = str(modelo)

                elif metodo == "Newton interpolante":
                    modelo = newton_interpolante(xs, ys)
                    tipo = "polinomio"
                    solucion_texto = str(modelo)

                elif metodo == "Lagrange":
                    modelo = lagrange_interpolante(xs, ys)
                    tipo = "polinomio"
                    solucion_texto = str(modelo)

                elif metodo == "Spline lineal":
                    modelo = spline_lineal(xs, ys)
                    tipo = "spline"
                    solucion_texto = texto_spline_lineal(xs, ys)

                else:
                    modelo = spline_cubico(xs, ys)
                    tipo = "spline"
                    solucion_texto = texto_spline_cubico(modelo, xs)

                st.write("Solución:")

                if metodo in ["Spline lineal", "Spline cúbico"]:
                    st.success(f"{metodo} generado correctamente.")
                    st.text(solucion_texto)
                else:
                    st.write(solucion_texto)

                fig = graficar_interpolacion(xs, ys, modelo, tipo)
                st.pyplot(fig)

                st.subheader("Comparación de error según porcentaje de datos almacenados")

                porcentajes = [10, 20, 30, 40]
                errores = []
                datos_usados = []

                for p in porcentajes:
                    k = max(2, int(np.ceil(len(xs) * p / 100)))

                    if metodo == "Spline cúbico":
                        k = max(4, k)

                    if k > len(xs):
                        errores.append(np.nan)
                        datos_usados.append(k)
                        continue

                    datos_usados.append(k)

                    indices = np.linspace(0, len(xs) - 1, k, dtype=int)
                    indices = np.unique(indices)

                    xs_sub = xs[indices]
                    ys_sub = ys[indices]

                    try:
                        if metodo == "Vandermonde":
                            modelo_sub = vandermonde(xs_sub, ys_sub)
                            y_pred = evaluar_polinomio(modelo_sub, xs)

                        elif metodo == "Newton interpolante":
                            modelo_sub = newton_interpolante(xs_sub, ys_sub)
                            y_pred = evaluar_polinomio(modelo_sub, xs)

                        elif metodo == "Lagrange":
                            modelo_sub = lagrange_interpolante(xs_sub, ys_sub)
                            y_pred = evaluar_polinomio(modelo_sub, xs)

                        elif metodo == "Spline lineal":
                            modelo_sub = spline_lineal(xs_sub, ys_sub)
                            y_pred = modelo_sub(xs)

                        else:
                            modelo_sub = spline_cubico(xs_sub, ys_sub)
                            y_pred = modelo_sub(xs)

                        error = np.linalg.norm(ys - y_pred)

                    except Exception:
                        error = np.nan

                    errores.append(error)

                df_error = pd.DataFrame({
                    "Porcentaje": porcentajes,
                    "Datos usados": datos_usados,
                    "Error": errores
                })

                st.dataframe(df_error)

                fig2, ax = plt.subplots()
                ax.plot(porcentajes, errores, marker="o")
                ax.set_xlabel("Porcentaje de datos almacenados")
                ax.set_ylabel("Error")
                ax.set_title(f"Comparación del error - {metodo}")
                ax.grid(True)
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"Error: {e}")