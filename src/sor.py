import src.utils as utils

def sor_iteration(A, b, eps, omega):
    """
    A Function to generate the solution of Ax=B via Successive Over Relaxation Iteration
    """
    row, col = len(A), len(A[0])
    solution_vector = utils.create_empty_vector(axis=1, length=max(row,col), fill=0)

    for r in range(max(row,col)):
        solution_vector[r] = [b[r][0] / A[r][r]]

    residual = utils.get_l2_norm_vector(
        utils.vector_difference(utils.multiply_matrices(A, solution_vector), b)
    )

    iterations = 0

    iter_list = []
    res_list = []
    while residual > eps:
        for r in range(row):
            sum_elem = b[r][0]
            for c in range(col):
                if r != c:
                    sum_elem -= A[r][c] * solution_vector[c][0]
            x_i = sum_elem / A[r][r]

            kth_iteration = solution_vector[r][0]
            solution_vector[r] = [((1 - omega) * kth_iteration) + (omega * x_i)]

        iterations += 1
        Ax = utils.multiply_matrices(A, solution_vector)
        residual = utils.get_l2_norm_vector(utils.vector_difference(Ax, b))

        iter_list.append(iterations)
        res_list.append(residual)
    return solution_vector, iter_list, res_list
