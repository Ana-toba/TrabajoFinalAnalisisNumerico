import numpy as np

def calcular_error(valor_actual, valor_anterior, tipo="absoluto"):
    if tipo == "relativo":
        if valor_actual == 0:
            return abs(valor_actual - valor_anterior)
        return abs((valor_actual - valor_anterior) / valor_actual)

    return abs(valor_actual - valor_anterior)

def norma_error(vector_actual, vector_anterior, tipo="absoluto"):
    if tipo == "relativo":
        denominador = np.linalg.norm(vector_actual)
        if denominador == 0:
            return np.linalg.norm(vector_actual - vector_anterior)
        return np.linalg.norm(vector_actual - vector_anterior) / denominador

    return np.linalg.norm(vector_actual - vector_anterior)
