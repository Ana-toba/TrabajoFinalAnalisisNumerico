import numpy as np

def _error(x_actual, x_anterior, A=None, b=None, tipo="absoluto"):
    if tipo == "relativo":
        denominador = np.linalg.norm(x_actual)
        if denominador == 0:
            return np.linalg.norm(x_actual - x_anterior)
        return np.linalg.norm(x_actual - x_anterior) / denominador

    if tipo == "condicion" and A is not None and b is not None:
        residuo = np.linalg.norm(b - A @ x_actual)
        condicion = np.linalg.cond(A)
        return condicion * residuo

    return np.linalg.norm(x_actual - x_anterior)

def jacobi(A, b, x0, tol, max_iter, tipo_error="absoluto"):
    D = np.diag(np.diag(A))
    L_U = A - D
    x_anterior = x0.copy()
    tabla = []

    for i in range(1, max_iter + 1):
        x_actual = np.linalg.inv(D) @ (b - L_U @ x_anterior)
        error = _error(x_actual, x_anterior, A, b, tipo_error)

        fila = {"Iteración": i, "Error": error}
        for j, valor in enumerate(x_actual):
            fila[f"x{j+1}"] = valor
        tabla.append(fila)

        if error < tol:
            return x_actual, tabla

        x_anterior = x_actual

    return x_actual, tabla

def gauss_seidel(A, b, x0, tol, max_iter, tipo_error="absoluto"):
    x_anterior = x0.copy()
    n = len(b)
    tabla = []

    for k in range(1, max_iter + 1):
        x_actual = x_anterior.copy()

        for i in range(n):
            suma1 = np.dot(A[i, :i], x_actual[:i])
            suma2 = np.dot(A[i, i+1:], x_anterior[i+1:])
            x_actual[i] = (b[i] - suma1 - suma2) / A[i, i]

        error = _error(x_actual, x_anterior, A, b, tipo_error)

        fila = {"Iteración": k, "Error": error}
        for j, valor in enumerate(x_actual):
            fila[f"x{j+1}"] = valor
        tabla.append(fila)

        if error < tol:
            return x_actual, tabla

        x_anterior = x_actual

    return x_actual, tabla

def sor(A, b, x0, w, tol, max_iter, tipo_error="absoluto"):
    x_anterior = x0.copy()
    n = len(b)
    tabla = []

    for k in range(1, max_iter + 1):
        x_actual = x_anterior.copy()

        for i in range(n):
            suma1 = np.dot(A[i, :i], x_actual[:i])
            suma2 = np.dot(A[i, i+1:], x_anterior[i+1:])
            nuevo = (b[i] - suma1 - suma2) / A[i, i]
            x_actual[i] = (1 - w) * x_anterior[i] + w * nuevo

        error = _error(x_actual, x_anterior, A, b, tipo_error)

        fila = {"Iteración": k, "Error": error}
        for j, valor in enumerate(x_actual):
            fila[f"x{j+1}"] = valor
        tabla.append(fila)

        if error < tol:
            return x_actual, tabla

        x_anterior = x_actual

    return x_actual, tabla

def radio_espectral_iteracion(A, metodo="jacobi", w=1.0):
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)

    if metodo == "jacobi":
        T = np.linalg.inv(D) @ (L + U)
    elif metodo == "gauss-seidel":
        T = np.linalg.inv(D - L) @ U
    elif metodo == "sor":
        T = np.linalg.inv(D - w * L) @ ((1 - w) * D + w * U)
    else:
        raise ValueError("Método no reconocido.")

    valores = np.linalg.eigvals(T)
    return max(abs(valores))
