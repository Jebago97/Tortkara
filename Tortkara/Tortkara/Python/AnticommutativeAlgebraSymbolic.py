import numpy as np
import sympy as sp

import BaseMath

class AnticommutativeAlgebraSymbolic:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.coefficients = np.ndarray((dimension, dimension, dimension), object)
        self.coefficients[:,:,:] = sp.Rational(0)

    def get_coefficient(self, i, j, k):
        return self.coefficients[i - 1, j - 1, k - 1]

