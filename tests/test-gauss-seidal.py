import src.utils as u
import src.gauss_seidel as gs

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
    
    b = [1] * dimension
    return matrix, b

if __name__ == "__main__":
    A_matrix, b_vector = generate_sparse_matrix_one_vector(1000)
    EPSILON = 1e-10

    gauss_solution_x, gauss_iter_list, gauss_res_list = gs.gauss_seidal_iteration(A_matrix, b_vector, eps=EPSILON)
    print("Gauss Seidel Iterations")
    print(f"L1-Norm of Solution Vector: {u.get_l1_norm_vector(gauss_solution_x)}")
    print(f"L2-Norm of Solution Vector: {u.get_l2_norm_vector(gauss_solution_x)}")
    print(f"L-Infinity-Norm of Solution Vector: {u.get_linf_norm_vector(gauss_solution_x)}")
    u.plot_graph("Gauss Seidel Iterations", "Iterations", "L-2 Residual", gauss_iter_list, gauss_res_list)
    
    vanilla_sol = u.vanilla_matrix_solution(A_matrix, b_vector)
    print("Vannila Iterations")
    print(f"L1-Norm of Solution Vector: {u.get_l1_norm_vector(vanilla_sol)}")
    print(f"L2-Norm of Solution Vector: {u.get_l2_norm_vector(vanilla_sol)}")
    print(f"L-Infinity-Norm of Solution Vector: {u.get_linf_norm_vector(vanilla_sol)}")
