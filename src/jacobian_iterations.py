import src.utils as utils

def jacobian_iteration(A, b, eps):
    """
    A Function to generate the solution of Ax=B via Jacobian Iteration
    """
    row, col = len(A), len(A[0])
    initial_guess = utils.create_empty_vector(axis=1, length=max(row,col), fill=0)
    
    for r in range(max(row, col)):
        initial_guess[r] = [b[r][0] / A[r][r]]
    solution_vector = utils.create_empty_vector(axis=1, length=max(row, col), fill=0)
    residual = utils.get_l2_norm_vector(
        utils.vector_difference(utils.multiply_matrices(A, initial_guess), b)
    )

    iterations = 0

    iter_list = []
    res_list = []
    while residual > eps:
        for r in range(row):
            sum_elem = b[r][0]
            for c in range(col):
                if r != c:
                    sum_elem -= A[r][c] * initial_guess[c][0]
            x_i = sum_elem / A[r][r]
            solution_vector[r] = [x_i]

        iterations += 1

        Ax = utils.multiply_matrices(A, solution_vector)
        residual = utils.get_l2_norm_vector(utils.vector_difference(Ax, b))

        iter_list.append(iterations)
        res_list.append(residual)

        initial_guess = solution_vector.copy()

    return solution_vector, iter_list, res_list