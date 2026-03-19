import numpy as np

def get_string_vector(vector: np.ndarray):
    if vector.ndim != 1:
        print('ERROR: el array debe tener dimension 1.')
        return ''
    if vector.size == 0:
        return '0'
    return " + ".join([f"{vector[i]}e_{i + 1}" for i in range(0, vector.size)])

def get_string_vector_no_zeros(vector: np.ndarray):
    if vector.ndim != 1:
        print('ERROR: el array debe tener dimension 1.')
        return ''
    s = " + ".join([f"{vector[i]}e_{i + 1}" for i in range(0, vector.size) if vector[i] != 0])
    if len(s) == 0:
        return '0'
    return s