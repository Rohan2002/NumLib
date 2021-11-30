import src.utils as u
import src.sor as sor
import tests.utils as u_t

if __name__ == "__main__":
    A_matrix, b_vector = u_t.generate_sparse_matrix_one_vector(5)
    EPSILON = 1e-10

    sor_solution_x, sor_iter_list, sor_res_list = sor.sor_iteration(A_matrix, b_vector, eps=EPSILON, omega=1.1)

    print("SOR Iterations")
    print(f"L1-Norm of Solution Vector: {u.get_l1_norm_vector(sor_solution_x)}")
    print(f"L2-Norm of Solution Vector: {u.get_l2_norm_vector(sor_solution_x)}")
    print(f"L-Infinity-Norm of Solution Vector: {u.get_linf_norm_vector(sor_solution_x)}")
    u.plot_graph("All Iterations", "Iterations", "L-2 Residual", sor_iter_list, sor_res_list)
    
    vanilla_sol = u.vanilla_matrix_solution(A_matrix, b_vector)
    print("Vannila Iterations")
    print(f"L1-Norm of Solution Vector: {u.get_l1_norm_vector(vanilla_sol)}")
    print(f"L2-Norm of Solution Vector: {u.get_l2_norm_vector(vanilla_sol)}")
    print(f"L-Infinity-Norm of Solution Vector: {u.get_linf_norm_vector(vanilla_sol)}")
