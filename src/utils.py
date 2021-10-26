import numpy as np
import matplotlib.pyplot as plt
import math
import os
"""
Some basic function for matrix operations and utility used in other modules
"""


def plot_graph(title, x_tit, y_tit, x_val, y_val):
    plt.title(title)
    plt.xlabel(x_tit, fontsize=14)
    plt.ylabel(y_tit, fontsize=14)
    plt.plot(x_val, y_val)
    title = "_".join( title.split() )
    plt.savefig(f"{os.path.dirname(os.getcwd())}/images/{title}.png")
    print("\n-------Created Graph---------")

def create_empty_vector(axis, length, fill=0):
    """
        Create a vector of 1xn row ector(axis = 0) or nx1 column vector(axis = 1) filled with fill arg.
    """
    if abs(axis) > 1:
        raise ValueError("Invalid axis...axis can only be 0(row) or 1(column")
    vector = [[fill]] * length if axis == 1 else [[fill] * length]
    return vector

def get_vector_element(vector: list, i:int):
    """
        Get a vector element at index i
    """
    row, col = get_row_col_matrix(vector)
    max_dim = max(row, col)
    if not is_vector(vector):
        raise ValueError("Not a vector!")
    if i < 0 or i >= max_dim:
        raise ValueError(f"Invalid index i {i}")

    return vector[i][0] if max_dim == row else vector[0][i]

def add_vector_element(vector: list, axis:int, element:int):
    """
        Added vector element based on axis=0 (1xn row vector) or axis=1 (nx1 column vector)
    """
    if not is_vector(vector):
        raise ValueError("Not a vector!")
    if axis == 1:
        vector.append([element])
    else:
        if len(vector) == 0:
            vector.append([element])
        else:
            vector[0].append(element)
    
def is_vector(vector):
    """
        Vector must be a 2-dimensional list.
    """
    if type(vector[0]) != list and type(vector[0]) != np.ndarray:
        raise ValueError(f"The given vector is 1-dimension {vector} and you must make it 2-dimensions!")
    row, col = get_row_col_matrix(vector)
    return min(row, col) == 1

def vector_difference(a, b):
    """
        Get the difference vector of a (dim:nx1) and b (dim:nx1) or a (dim:1xn) and b (dim:1xn)
    """
    row_a, col_a = get_row_col_matrix(a)
    row_b, col_b = get_row_col_matrix(b)
    
    if not is_vector(a) or not is_vector(b):
        raise ValueError("a and b must be vectors!")
    
    if ((row_a != row_b) or (col_a != col_b)):
        raise ValueError("Vectors a and b can be either nx1 and nx1 resp. or 1xn and 1xn respectivily")

    # It doesn't matter max(row_a, col_a) or max(row_b, col_b) because row_a must equal row_b and col_a must equal col_b.
    is_row_vector = max(row_a, col_a) == col_a
    diff = create_empty_vector(axis=0, length=col_a, fill=0) if is_row_vector else create_empty_vector(axis=1, length=row_a, fill=0)
    for i in range(max(row_a, col_a)):
        a_element = get_vector_element(a, i)
        b_element = get_vector_element(b, i)
        if is_row_vector:
            diff[0][i] = abs(a_element - b_element)
        else:
            diff[i] = [abs(a_element - b_element)]
    return diff


def get_l1_norm_vector(vector):
    """
    A function to get L1-Norm of Vector V. If V = [x1, x2], norm(V) = |x1| + |x2|
    """
    row, col = get_row_col_matrix(vector)
    norm = 0
    for i in range(0, max(row, col)):
        # Do addition based on nx1 or 1xn vector
        vec_elem = get_vector_element(vector, i)
        norm += vec_elem
    return norm


def get_l2_norm_vector(vector):
    """
    A function to get L2-Norm of Ve ctor V. If V = [x1, x2], norm(V) = sqrt((x1)^2) + (x2)^2)
    """
    row, col = get_row_col_matrix(vector)
    norm = 0
    for i in range(0, max(row, col)):
        # Do Squaring based on nx1 or 1xn vector
        vec_elem = get_vector_element(vector, i)
        norm += vec_elem ** 2
    return math.sqrt(norm)


def get_linf_norm_vector(vector):
    """
    A function to get Linfinity-Norm of Vector V. If V = [x1, x2], norm(V) = max(|x1|, |x2|)
    """
    row, col = get_row_col_matrix(vector)
    norm = 0
    for i in range(0, max(row, col)):
        # Do comparison based on nx1 or 1xn vector
        vec_elem = get_vector_element(vector, i)
        if abs(vec_elem) > norm:
            norm = abs(vec_elem)
    return norm


def get_row_col_matrix(matrix):
    """
    This function will return the number of row and col as a tuple based on the type of matrix.
    Since Numpy matrices, and python list have different ways to extract the number of row and column,
    we need a robust way to extract the row and column based on the library of the matrix.
    """
    row_length, col_length = 0, 0
    if type(matrix) == np.ndarray:
        row_length, col_length = matrix.shape

    if type(matrix) == list:
        row_length, col_length = len(matrix), len(matrix[0])
    return row_length, col_length


def generate_square_identity_matrix(dim):
    """
    This function will return a nxn identity matrix
    """
    matrix = []
    for row in range(0, dim):
        column = []
        for col in range(0, dim):
            if row == col:  # Append 1 in the diagnols only
                column.append(1)
            else:  # Else append 0 in all other positions in the matrix
                column.append(0)
        matrix.append(column)
    return matrix

def transpose_vector(vector):
    """
    This function is made to transpose a nx1 vector to 1xn or 1xn vector to nx1. It's used in the forward and backward
    substitution because the solution vector is initially a python list (1xn vector). However, to make the matrix multiplication
    compatible with matrix A(nxn dimension) and B(nx1 vector), the solution vector x must be a nx1 vector too. Hence the equation,
    Ax=B will be satisified as the dimensions have been fixed.
    """
    row, col = len(vector), len(vector[0])

    # 1x6 = 6 or 6x1 = 6
    if row * col != max(row, col):
        raise ValueError("Arguement must be a vector!")

    # If we have a nx1 vector, then it will be shaped as 1xn vector here.
    if row > col:
        col = []
        for r in range(0, row):
            col.append(vector[r][0])
        return [col]

    # If we have a 1xn vector, then it will be shaped as nx1 vector here.
    if col > row:
        row = []
        for c in range(0, col):
            row.append([vector[0][c]])
        return row


def round_matrix(matrix, round_value=4):
    """
    Round matrices to a certain decimal place
    """
    row_length, col_length = get_row_col_matrix(matrix)
    for r in range(0, row_length):
        for c in range(0, col_length):
            matrix[r][c] = round(
                matrix[r][c], round_value
            )  # Round each value of the matrix and replace the original value with the rounded value.
    return matrix


def get_column(matrix, i):
    """
    This function is a helper method to get the columns for multiplication in multiply_matrices(A, B) method.
    """
    if i >= len(matrix):
        raise ValueError(f"Column Index greater than the number of columns!")
    return [[
        row[i] for row in matrix
    ]]  # Returns the column of the matrix as a python list

def vector_dot_product(A, B):
    """
    A function to that returns a scalar dot product of nx1 vectors.
    """
    row_length, col_length = get_row_col_matrix(A)
    dot_product_sum = 0
    for i in range(0, max(row_length, col_length)):
        a_element = get_vector_element(A, i)
        b_element = get_vector_element(B, i)
        dot_product_sum += (
            a_element * b_element
        )  # Go through all elements of both vectors A, B and multiply them. Finally sum of all the multiplications to get dot product.
    return dot_product_sum


def multiply_matrices(A, B):
    """
    A function to multiply two matrices of size nxm and mxn and vice versa.
    """
    A_row, A_col = len(A), len(A[0])
    B_row, B_col = len(B), len(B[0])
    product_matrix = []

    if A_col != B_row:
        raise ValueError(f"Can't multiply matrices as their dimensions don't match!")
    for row in range(0, A_row):
        A_row_vector = [A[row]]  # Get the row of matrix 1
        product_matrix_col = []
        for col in range(0, B_col):
            B_col_vector = get_column(B, col)  # Get column of matrix 2
            vector_dot_product_value = vector_dot_product(
                A_row_vector, B_col_vector
            )  # Find the dot product of the row of matrix 1 and col of matrix 2 to get a scalar
            product_matrix_col.append(
                vector_dot_product_value
            )  # Append the multiplication result to the new column

        product_matrix.append(
            product_matrix_col
        )  # Append column to a 2-d list to form rows of columns
    return product_matrix


def print_matrix(matrix):
    """
    A simple function made to print the matrix in a readable form.
    """
    row_length, col_length = get_row_col_matrix(matrix)

    for row in range(0, row_length):
        col_string = "["
        for col in range(0, col_length):
            col_string += f"{matrix[row][col]},"
        col_string += "]"
        print(col_string.strip())


def calculate_l2_relative_error(original_vector, new_vector):
    """
    A function to get calculate the relative error of the norms of two vectors
    """
    norm_x = get_l2_norm_vector(original_vector)
    norm_lu = get_l2_norm_vector(new_vector)

    return round(abs(norm_x - norm_lu) / abs(norm_x), 4)


def vanilla_matrix_solution(A, B):
    """
    This is the vanilla method to calculate x for the equation Ax = B
    Method: x = Inverse(A) * B

    This purely using numpy! But very inefficient for bigger matrix dimensions.
    """
    A_numpy = np.asarray(A, dtype=np.float64)
    b_numpy = np.asarray(B, dtype=np.float64)

    return np.around(np.dot(np.linalg.inv(A_numpy), b_numpy), decimals=4)
