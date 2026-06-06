import numpy as np
import sympy as sp

import BaseMath

class AnticommutativeAlgebraSymbolic:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.coefficients = sp.MutableDenseNDimArray([0]*(dimension**3),(dimension, dimension, dimension))

    def get_coefficient(self, i, j, k):
        return self.coefficients[i - 1, j - 1, k - 1]

    def set_coefficient_value(self, i, j, k, value):
        if i == j and value != 0:
            print(f"The product e_{i} e_{j} must be zero.")
            return
        i0 = i - 1
        j0 = j - 1
        k0 = k - 1
        self.coefficients[i0, j0, k0] = value
        self.coefficients[j0, i0, k0] = -value

    #Edge from i to j
    def add_directed_edge_value(self, i, j, weight):
        if weight == 0:
            print("ERROR: The weight is zero.")
            return
        if i<j:
            self.set_coefficient_value(i, j, j, weight)
        elif i>j:
            self.set_coefficient_value(j, i, j, weight)

    def add_triangle_value(self, i, j, k, weight_ij, weight_ik, weight_jk):
        if weight_ij == 0 and weight_ik == 0 and not weight_jk == 0:
            print("All the edges of a triangle can not be zero.")
            return
        if i < j:
            self.set_coefficient_value(i, j, k, weight_ij)
        elif i > j:
            self.set_coefficient_value(j, i, k, weight_ij)
        if i < k:
            self.set_coefficient_value(i, k, j, weight_ik)
        elif i > k:
            self.set_coefficient_value(k, i, j, weight_ik)
        if j < k:
            self.set_coefficient_value(j, k, i, weight_jk)
        elif j > k:
            self.set_coefficient_value(k, j, i, weight_jk)

    def show_products(self):
        n = self.dimension
        for i in range(0, n):
            for j in range(0, n):
                print(f"e_{i + 1}e_{j + 1}: {BaseMath.get_string_vector_no_zeros(self.coefficients[i, j])}")

    def get_base_vectors(self):
        return BaseMath.get_base_vectors(self.dimension)

    # ----------------------------------------

    def bilineal_vector_product_int_int(self, i: int, j: int):
        return self.coefficients[i, j]

    def bilineal_vector_product_vector_int(self, v1: sp.MutableDenseNDimArray, j: int):
        v = sp.MutableDenseNDimArray([0]*self.dimension)
        for i in range(0, self.dimension):
            v += v1[i]*self.coefficients[i, j]
        return v

    def bilineal_vector_product_int_vector(self, i: int, v2: sp.MutableDenseNDimArray):
        v = sp.MutableDenseNDimArray([0] * self.dimension)
        for j in range(0, self.dimension):
            v += v2[j] * self.coefficients[i, j]
        return v

    def bilineal_vector_product(self, v1: sp.MutableDenseNDimArray, v2: sp.MutableDenseNDimArray):
        v = sp.MutableDenseNDimArray([0] * self.dimension)
        for i in range(0, self.dimension):
            for j in range(0, self.dimension):
                v += v1[i]*v2[j]*self.coefficients[i, j]
        return v

    def bilineal_vector_product_vector_int_einsum(self, v1, j: int):
        return sp.MutableDenseNDimArray(np.einsum('i,ik->k', v1, self.coefficients[:, j]))

    def bilineal_vector_product_int_vector_einsum(self, i: int, v2):
        return sp.MutableDenseNDimArray(np.einsum('jk,j->k', self.coefficients[i], v2))

    def bilineal_vector_product_einsum(self, v1, v2):
        return sp.MutableDenseNDimArray(np.einsum('i,ijk,j->k', v1, self.coefficients, v2))

    # ----------------------------------------

    def jacobi_identity(self, i: int, j: int, k: int, simplify: bool = True) -> sp.MutableDenseNDimArray:
        r1 = self.bilineal_vector_product_vector_int(self.bilineal_vector_product_int_int(i, j), k)
        r2 = self.bilineal_vector_product_vector_int(self.bilineal_vector_product_int_int(j, k), i)
        r3 = self.bilineal_vector_product_vector_int(self.bilineal_vector_product_int_int(k, i), j)
        jacobi_identity = r1 + r2 + r3
        if simplify:
            for index in range(0, self.dimension):
                jacobi_identity[index] = sp.expand(sp.simplify(jacobi_identity[index]))
        return jacobi_identity

    def tortkara_identity(self, i: int, j: int, k: int, l: int, simplify: bool = True) -> sp.MutableDenseNDimArray:
        left_side = self.tortkara_identity_left_side(i, j, k, l) + self.tortkara_identity_left_side(i, l, k, j)
        right_side = self.tortkara_identity_right_side(i, j, k, l) + self.tortkara_identity_right_side(i, l, k, j)
        tortkara_identity = left_side - right_side
        if simplify:
            for index in range(0, self.dimension):
                tortkara_identity[index] = sp.expand(sp.simplify(tortkara_identity[index]))
        return tortkara_identity

    def tortkara_identity_left_side(self, i: int, j: int, k: int, l: int):
        return self.bilineal_vector_product(self.bilineal_vector_product_int_int(i, j), self.bilineal_vector_product_int_int(k, l))

    def tortkara_identity_right_side(self, i: int, j: int, k: int, l: int):
        return self.bilineal_vector_product_vector_int(self.jacobi_identity(i,j,k,False), l)

    # def tortkara_identity_right_side(self, i: int, j: int, k: int, l: int):
    #     r1 = self.bilineal_vector_product_vector_int(self.bilineal_vector_product_int_int(i, j), k)
    #     r2 = self.bilineal_vector_product_vector_int(self.bilineal_vector_product_int_int(j, k), i)
    #     r3 = self.bilineal_vector_product_vector_int(self.bilineal_vector_product_int_int(k, i), j)
    #     jacobi_ijk = r1 + r2 + r3
    #     return self.bilineal_vector_product_vector_int(jacobi_ijk, l)

    # ----------------------------------------

    def is_lie_algebra(self):
        dim = self.dimension
        is_lie = True
        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    identity = self.jacobi_identity(i, j, k)
                    if not np.any(identity):
                        continue
                    if is_lie:
                        is_lie = False
                    print(f"e_{i}, e_{j}, e_{k}: {BaseMath.get_string_vector(identity)} != 0")
        if is_lie:
            print("It's a Lie Algebra, yay!")
        else:
            print("It's NOT a Lie Algebra :(")
        return is_lie

    def is_tortkara_algebra(self):
        dim = self.dimension
        is_tortkara = True
        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    for l in range(0, dim):
                        identity = self.tortkara_identity(i, j, k, l)
                        if not np.any(identity):
                            continue
                        if is_tortkara:
                            is_tortkara = False
                        print(f"e_{i}, e_{j}, e_{k}, e_{l}: {BaseMath.get_string_vector(identity)} != 0")
        if is_tortkara:
            print("It's a Tortkara Algebra, yay!")
        else:
            print("It's NOT a Tortkara Algebra :(")
        return is_tortkara

    def derived_series(self):
        # A_n = A_(n-1)A_(n-1)
        base_vectors = self.get_base_vectors()
        k = 1
        prev_a_n = None
        a_n = base_vectors
        print(f"Space of A_{k}: ")
        BaseMath.print_generators(a_n)
        k += 1
        while len(a_n) > 0:
            prev_a_n = a_n
            n_elements = len(prev_a_n)
            list_generators = []
            for i in range(0, n_elements):
                for j in range(0, n_elements):
                    list_generators.append(self.bilineal_vector_product(prev_a_n[i], prev_a_n[j]))
            matrix_generators = sp.Matrix(list_generators)
            matrix_generators, pivots = matrix_generators.rref()
            a_n = [sp.MutableDenseNDimArray(matrix_generators)[i, :] for i in range(0, matrix_generators.rows) if
                   np.any(sp.MutableDenseNDimArray(matrix_generators)[i, :])]
            print("")
            print(f"Space of A_{k}: ")
            if len(a_n) == len(prev_a_n):
                print(f"A_{k} = A_{k-1}")
                break
            BaseMath.print_generators(a_n)
            k += 1

    def central_series(self):
        #A^n = A^(n-1)A
        base_vectors = self.get_base_vectors()
        k = 1
        prev_a_n = None
        a_n = base_vectors
        print(f"Space of A^{k}: ")
        BaseMath.print_generators(a_n)
        k += 1
        while len(a_n) > 0:
            prev_a_n = a_n
            n_elements = len(prev_a_n)
            list_generators = []
            for i in range(0,n_elements):
                for j in range(0,n_elements):
                    list_generators.append(self.bilineal_vector_product(prev_a_n[i], base_vectors[j]))
            matrix_generators = sp.Matrix(list_generators)
            matrix_generators, pivots = matrix_generators.rref()
            a_n = [sp.MutableDenseNDimArray(matrix_generators)[i,:] for i in range(0,matrix_generators.rows) if
                   np.any(sp.MutableDenseNDimArray(matrix_generators)[i,:])]
            print("")
            print(f"Space of A^{k}: ")
            if len(a_n) == len(prev_a_n):
                print(f"A^{k} = A^{k-1}")
                break
            BaseMath.print_generators(a_n)
            k += 1
