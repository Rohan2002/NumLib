import src.utils as vector_utils

def is_square(mat):
    return len(mat) == len(mat[0])

def round_matrix(matrix, round_value=4):
    """
    Round matrices to a certain decimal place
    """
    row_length, col_length = len(matrix), len(matrix[0])
    for r in range(0, row_length):
        for c in range(0, col_length):
            matrix[r][c] = round(
                matrix[r][c], round_value
            )  # Round each value of the matrix and replace the original value with the rounded value.
    return matrix

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

# TODO: Create a function to check if nxn matrix is invertible or not. (Check if RANK(mat) == n)
def get_matrix_inverse(mat_input):
    """
        Suppose matrix A is a nxn invertible matrix, then A^-1 is the inverse of matrix A iff A*A^-1 = 0
    """
    mat = mat_input.copy()
    if not is_square(mat):
        raise ValueError("The given matrix is not a square matrix!")
    
    n = len(mat)
    N = generate_square_identity_matrix(n)
    
    for p in range(n):
        f = mat[p][p]
        for c in range(n):
            mat[p][c] /= f
            N[p][c] /= f
        for i in range(p+1, n):
            f = mat[i][p]
            for c in range(n):
                mat[i][c] -= (mat[p][c] * f)
                N[i][c] -= (N[p][c] * f)
    for p in range(n-1, -1, -1):
        for i in range(p-1, -1, -1):
            f = mat[i][p]
            for c in range(n):
                mat[i][c] -= (mat[p][c] * f)
                N[i][c] -= (N[p][c] * f)
    return N

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
            B_col_vector = vector_utils.get_column(B, col)  # Get column of matrix 2
            vector_dot_product_value = vector_utils.vector_dot_product(
                A_row_vector, B_col_vector
            )  # Find the dot product of the row of matrix 1 and col of matrix 2 to get a scalar
            product_matrix_col.append(
                vector_dot_product_value
            )  # Append the multiplication result to the new column

        product_matrix.append(
            product_matrix_col
        )  # Append column to a 2-d list to form rows of columns
    return product_matrix

