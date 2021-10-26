import src.utils as u
import src.jacobian_iterations as jac

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

if __name__ == "__main__":
    A_matrix, b_vector = generate_sparse_matrix_one_vector(5)
    EPSILON = 1e-10

    jacobian_solution_x, jac_iter_list, jac_res_list = jac.jacobian_iteration(A_matrix, b_vector, eps=EPSILON)
    print("Jacobian Iterations")
    print(f"L1-Norm of Solution Vector: {u.get_l1_norm_vector(jacobian_solution_x)}")
    print(f"L2-Norm of Solution Vector: {u.get_l2_norm_vector(jacobian_solution_x)}")
    print(f"L-Infinity-Norm of Solution Vector: {u.get_linf_norm_vector(jacobian_solution_x)}")
    u.plot_graph("Jacobian Iterations", "Iterations", "L-2 Residual", jac_iter_list, jac_res_list)
    print("\n")

    vanilla_sol = u.vanilla_matrix_solution(A_matrix, b_vector)
    print("Vannila Iterations")
    print(f"L1-Norm of Solution Vector: {u.get_l1_norm_vector(vanilla_sol)}")
    print(f"L2-Norm of Solution Vector: {u.get_l2_norm_vector(vanilla_sol)}")
    print(f"L-Infinity-Norm of Solution Vector: {u.get_linf_norm_vector(vanilla_sol)}")
