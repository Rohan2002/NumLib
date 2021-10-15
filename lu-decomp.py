import numpy as np
def vanilla_matrix_solution(A, B):
    """
    This is the vanilla method to calculate x for the equation Ax = B
    Method: x = Inverse(A) * B

    This purely using numpy!
    """
    A_numpy = np.asarray(A, dtype=np.float64)
    b_numpy = np.asarray(B, dtype=np.float64)

    return np.around(np.dot(np.linalg.inv(A_numpy), b_numpy), decimals=4)


"""
    Down below is the beginning of the implementation of LU-Decomposition to solve the equation Ax=B. It's purely made
    without using external libraries and everything is made from scratch.

    Author: Rohan Deshpande
"""


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


def reshape_vector(vector):
    """
    This function is made to reshape a nx1 vector to 1xn or 1xn vector to nx1. It's used in the forward and backward
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
    return [
        row[i] for row in matrix
    ]  # Returns the column of the matrix as a python list


def vector_dot_product(A, B):
    """
    A function to that returns a scalar dot product of nx1 vectors.
    """
    if len(A) != len(B):
        raise ValueError(f"Can't do dot product of vectors!")
    dot_product_sum = 0
    for i in range(0, len(A)):
        dot_product_sum += (
            A[i] * B[i]
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
        A_row_vector = A[row]  # Get the row of matrix 1
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


def get_lower_upper_triangular_matrices(A):
    """
    LU-Decomposition: Decompose nxn matrix into Lower and Upper Triangular Matrices
    """
    row, col = len(A), len(A[0])
    lower_triangular = generate_square_identity_matrix(row)
    upper_triangular = A

    for r in range(0, row):
        diagnol_element = upper_triangular[r][r]  # This is the element in the diagnol
        for c in range(r + 1, col):
            elements_under_diagnol = upper_triangular[c][r]
            ratio = -(elements_under_diagnol / diagnol_element)

            temp_inverse = generate_square_identity_matrix(row)
            temp_inverse[c][r] = -ratio

            lower_triangular = multiply_matrices(lower_triangular, temp_inverse)
            for col_value in range(0, len(upper_triangular[0])):
                upper_triangular[c][col_value] = upper_triangular[c][col_value] + (
                    upper_triangular[r][col_value] * ratio
                )

    return round_matrix(lower_triangular, round_value=4), round_matrix(
        upper_triangular, round_value=4
    )


def backward_subsitution(A, b):
    """
    Backward substitution solves the matrix equation Ax=B and returns the solution vector x.
    Condition: A must be a nxn upper triangular matrix, and B = nx1 vector
    """
    solution_vector = []

    n = len(A) - 1  # Calculate n
    x_n = (
        b[n][0] / A[n][n]
    )  # Calculate the first solution, such that it can be used to solve other solutions.
    solution_vector.append(x_n)  # Append first solution

    for i in range(n - 1, -1, -1):
        sum_elements_after_diag = 0
        for elem_index in range(i + 1, n + 1):
            sum_elements_after_diag += (
                A[i][elem_index] * solution_vector[n - elem_index]
            )  # Get the sum of the row by multiplying the coefficient with the cached solution vector to get the solution x_i
        x_i = (b[i][0] - sum_elements_after_diag) / A[i][
            i
        ]  # Finalize x_i by subtracting previous sum from the RHS of linear equation b_i and dividing by the coefficient of x_i to get the final x_i solution

        solution_vector.append(x_i)  # Append solution x_i

    return reshape_vector([solution_vector[::-1]])  # Change 1xn vector to nx1 vector.


def forward_subsitution(A, b):
    """
    Forward substitution solves the matrix equation Ax=B and returns the solution vector x.
    Condition: A must be a nxn lower triangular matrix, and B = nx1 vector
    """
    solution_vector = []

    n = len(A)  # Calculate n
    x_n = (
        b[0][0] / A[0][0]
    )  # Calculate the first solution, such that it can be used to solve other solutions.
    solution_vector.append(x_n)  # Append first solution

    for i in range(1, n):
        sum_elements_after_diag = 0
        for elem_index in range(0, i):
            sum_elements_after_diag += (
                A[i][elem_index] * solution_vector[elem_index]
            )  # Get the sum of the row by multiplying the coefficient with the cached solution vector to get the solution x_i
        x_i = (b[i][0] - sum_elements_after_diag) / A[i][
            i
        ]  # Finalize x_i by subtracting previous sum from the RHS of linear equation b_i and dividing by the coefficient of x_i to get the final x_i solution

        solution_vector.append(x_i)  # Append solution x_i

    return reshape_vector([solution_vector])  # Change 1xn vector to nx1 vector.


def lu_matrix_solution(A, B):
    """
    Solve Ax=B Using LU-Decomponsition
    Since A = LU, then LUx = B.

    Assume Ux = Y, then LY = B.
    We can solve for Y using forward substitution.

    Once we have Y solution vector, We can solve for x
    through Ux=Y. Hence we have the solution x for Ax=B
    """
    lower_triangular, upper_triangular = get_lower_upper_triangular_matrices(
        A
    )  # Get lower and upper_triangular matrices

    Y = forward_subsitution(lower_triangular, B)  # LY = B, and we get Y.
    X = backward_subsitution(
        upper_triangular, Y
    )  # Using Y and U, solve UX =Y for solution vector X

    return round_matrix(X, 4)


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


def get_l2_norm_vector(vector):
    """
    A function to get L2-Norm of Ve ctor V. If V = [x1, x2], norm(V) = sqrt((x1)^2) + (x2)^2)
    """
    row, col = get_row_col_matrix(vector)
    norm = 0
    for i in range(0, max(row, col)):
        # Do Squaring based on nx1 or 1xn vector
        if row > col:
            norm += vector[i][0] ** 2
        else:
            norm += vector[0][i] ** 2
    return np.sqrt(norm)


def calculate_l2_relative_error(original_vector, new_vector):
    """
    A function to get calculate the relative error of the norms of two vectors
    """
    norm_x = get_l2_norm_vector(original_vector)
    norm_lu = get_l2_norm_vector(new_vector)

    return round(abs(norm_x - norm_lu) / abs(norm_x), 4)


if __name__ == "__main__":

    A = [
        [21, 32, 14, 8, 6, 9, 11, 3, 5],
        [17, 2, 8, 14, 55, 23, 19, 1, 6],
        [41, 23, 13, 5, 11, 22, 26, 7, 9],
        [12, 11, 5, 8, 3, 15, 7, 25, 19],
        [14, 7, 3, 5, 11, 23, 8, 7, 9],
        [2, 8, 5, 7, 1, 13, 23, 11, 17],
        [11, 7, 9, 5, 3, 8, 26, 13, 17],
        [23, 1, 5, 19, 11, 7, 9, 4, 16],
        [31, 5, 12, 7, 13, 17, 24, 3, 11],
    ]
    B = [[2], [5], [7], [1], [6], [9], [4], [8], [3]]

    vanilla_x_solution = vanilla_matrix_solution(A, B)
    lu_x_solution = lu_matrix_solution(A, B)

    vanilla_row, vanilla_col = get_row_col_matrix(vanilla_x_solution)
    print(
        f"The solution vector using the vanilla approach of Inverse(A) * B = X with dimensions {vanilla_row}x{vanilla_col} using numpy library:"
    )
    print_matrix(vanilla_x_solution)

    print("\n")

    lu_row, lu_col = get_row_col_matrix(lu_x_solution)
    print(
        f"The solution vector using the LU-Decomposition, Forward Substitution and Backward Substitution to find solution vector x with dimensions {lu_row}x{lu_col} from scratch:"
    )
    print_matrix(lu_x_solution)

    print("\n")

    relative_error = calculate_l2_relative_error(vanilla_x_solution, lu_x_solution)
    print(f"The relative error between X and Xlu is {relative_error * 100}%")
