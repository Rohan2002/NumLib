import src.utils as u

def generate_sparse_matrix_one_vector(dimension):
    """
        A function to generate a sparse matrix of dimension nxn and a vector of all ones.
    """
    matrix = []
    for row in range(dimension):
        row_vector = [0] * dimension # Init a sparse row
        for col in range(len(row_vector)):
            if row == col:
                row_vector[row] = 40
                if row > 0:
                    row_vector[row-1] = -10
                if row < len(row_vector) - 1:
                    row_vector[row+1] = -10
        matrix.append(row_vector)
    
    b = u.create_empty_vector(axis = 1, length=dimension, fill=1)
    return matrix, b