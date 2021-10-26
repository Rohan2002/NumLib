import src.utils as utils
"""
    Down below is the beginning of the implementation of LU-Decomposition to solve the equation Ax=B. It's purely made
    without using external libraries and everything is made from scratch.

    Author: Rohan Deshpande
"""

def get_lower_upper_triangular_matrices(A):
    """
    LU-Decomposition: Decompose nxn matrix into Lower and Upper Triangular Matrices
    """
    row, col = len(A), len(A[0])
    lower_triangular = utils.generate_square_identity_matrix(row)
    upper_triangular = A

    for r in range(0, row):
        diagnol_element = upper_triangular[r][r]  # This is the element in the diagnol
        for c in range(r + 1, col):
            elements_under_diagnol = upper_triangular[c][r]
            ratio = -(elements_under_diagnol / diagnol_element)

            temp_inverse = utils.generate_square_identity_matrix(row)
            temp_inverse[c][r] = -ratio

            lower_triangular = utils.multiply_matrices(lower_triangular, temp_inverse)
            for col_value in range(0, len(upper_triangular[0])):
                upper_triangular[c][col_value] = upper_triangular[c][col_value] + (
                    upper_triangular[r][col_value] * ratio
                )

    return utils.round_matrix(lower_triangular, round_value=4), utils.round_matrix(
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

    return utils.transpose_vector([solution_vector[::-1]])  # Change 1xn vector to nx1 vector.


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

    return utils.transpose_vector([solution_vector])  # Change 1xn vector to nx1 vector.


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

    return utils.round_matrix(X, 4)
