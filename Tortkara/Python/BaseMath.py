import numpy as np
import sympy as sp

def get_string_vector(array: sp.tensor.array.dense_ndim_array.NDimArray):
    if array.rank() != 1:
        print('ERROR: array is not 1-dimensional.')
        return ''
    return " + ".join([f"{array[i]}e_{i + 1}" for i in range(0, array.shape[0])])

def get_string_vector_no_zeros(array: sp.tensor.array.dense_ndim_array.NDimArray):
    if array.rank() != 1:
        print('ERROR: array is not 1-dimensional.')
        return ''
    s = " + ".join([f"{array[i]}e_{i + 1}" for i in range(0, array.shape[0]) if array[i] != 0])
    if len(s) == 0:
        return '0'
    return s

def get_base_vectors(n):
    base_vectors = []
    for i in range(0, n):
        e_i = sp.MutableDenseNDimArray([0] * n)
        e_i[i] = 1
        base_vectors.append(e_i)
    return base_vectors

def print_generators(list_generators):
    l = len(list_generators)
    if l == 0:
        print("0")
    for i in range(0, l):
        print(get_string_vector_no_zeros(list_generators[i]))
