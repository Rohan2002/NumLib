import math
import matplotlib.pyplot as plt

def generate_A_matrix_b_vector(dimension):
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

def plot_graph(title, x_tit, y_tit, x_val, y_val):
    plt.title(title)
    plt.xlabel(x_tit, fontsize=14)
    plt.ylabel(y_tit, fontsize=14)
    plt.plot(x_val, y_val)
    plt.savefig(f'./{title}.png')
    print("\n-------Created Graph---------")

def dot_matrix_vector(mat, vec):
    if len(mat[0]) != len(vec):
        raise ValueError("Invalid Mulitplication Dimension")
    
    solution_vector = [0] * len(vec)
    for row in range(len(mat)):
        row_vector = mat[row]       
        sum = 0
        for elem in range(len(row_vector)):
            sum += (row_vector[elem] * vec[elem])
        solution_vector[row] = sum
    return solution_vector


def vector_difference(a, b):
    
    if len(a) != len(b):
        raise ValueError("Vectors must be same dimensions!")
    diff = []
    for i in range(len(a)):
        diff.append(abs(a[i] - b[i]))
    return diff

def get_l1_norm_vector(vector):
    """
    A function to get L1-Norm of Vector V. If V = [x1, x2], norm(V) = |x1| + |x2|
    """
    norm = 0
    for i in range(0, len(vector)):
        norm += abs(vector[i])
    return norm

def get_l2_norm_vector(vector):
    """
    A function to get L2-Norm of Vector V. If V = [x1, x2], norm(V) = sqrt((x1)^2) + (x2)^2)
    """
    norm = 0
    for i in range(0, len(vector)):
        norm += vector[i] ** 2
    return math.sqrt(norm)

def get_linf_norm_vector(vector):
    """
    A function to get Linfinity-Norm of Vector V. If V = [x1, x2], norm(V) = max(|x1|, |x2|)
    """
    norm = 0
    for i in range(0, len(vector)):
        if abs(vector[i]) > norm:
            norm = abs(vector[i])
    return norm

def jacobian_iteration(A, b, eps):
    """
        A Function to generate the solution of Ax=B via Jacobian Iteration
    """
    row, col = len(A), len(A[0])
    initial_guess = [0] * max(row, col)
    
    for r in range(row):
        initial_guess[r] = b[r] / A[r][r]
    
    solution_vector = [0] * max(row, col)
    residual = get_l2_norm_vector(vector_difference(dot_matrix_vector(A, initial_guess), b))

    iterations = 0

    iter_list = []
    res_list = []
    while residual > eps:
        for r in range(row):
            sum_elem = b[r]
            for c in range(col):
                if r != c:
                    sum_elem -= (A[r][c] * initial_guess[c])
            x_i = sum_elem / A[r][r]
            solution_vector[r] = x_i
        
        iterations +=1
        
        Ax = dot_matrix_vector(A, solution_vector)
        residual = get_l2_norm_vector(vector_difference(Ax, b))
        
        iter_list.append(iterations)
        res_list.append(residual)

        initial_guess = solution_vector.copy()
    
    return solution_vector, iter_list, res_list


def sor_iteration(A, b, eps, omega):
    """
        A Function to generate the solution of Ax=B via Successive Over Relaxation Iteration
    """
    row, col = len(A), len(A[0])
    solution_vector = [0] * max(row, col)

    for r in range(row):
        solution_vector[r] = b[r] / A[r][r]
    
    residual = get_l2_norm_vector(vector_difference(dot_matrix_vector(A, solution_vector), b))

    iterations = 0

    iter_list = []
    res_list = []
    while residual > eps:
        for r in range(row):
            sum_elem = b[r]
            for c in range(col):
                if r != c:
                    sum_elem -= (A[r][c] * solution_vector[c])
            x_i = sum_elem / A[r][r]
            
            kth_iteration = solution_vector[r]
            solution_vector[r] = ((1-omega) * kth_iteration) + (omega * x_i)
        
        iterations +=1
        Ax = dot_matrix_vector(A, solution_vector)
        residual = get_l2_norm_vector(vector_difference(Ax, b))
        
        iter_list.append(iterations)
        res_list.append(residual)
    return solution_vector, iter_list, res_list

def print_matrix(matrix):
    """
    A simple function made to print the matrix in a readable form.
    """
    for row in range(len(matrix)):
        print(matrix[row])

if __name__ == "__main__":
    A_matrix, b_vector = generate_A_matrix_b_vector(1000)
    EPSILON = 1e-10

    jacobian_solution_x, jac_iter_list, jac_res_list = jacobian_iteration(A_matrix, b_vector, eps=EPSILON)
    gauss_solution_x, gauss_iter_list, gauss_res_list = sor_iteration(A_matrix, b_vector, eps=EPSILON, omega=1)
    sor_solution_x, sor_iter_list, sor_res_list = sor_iteration(A_matrix, b_vector, eps=EPSILON, omega=1.1)
    
    print("Jacobian Iterations")
    print(f"L1-Norm of Solution Vector: {get_l1_norm_vector(jacobian_solution_x)}")
    print(f"L2-Norm of Solution Vector: {get_l2_norm_vector(jacobian_solution_x)}")
    print(f"L-Infinity-Norm of Solution Vector: {get_linf_norm_vector(jacobian_solution_x)}")
    plot_graph("Jacobian Iterations", "Iterations", "L-2 Residual", jac_iter_list, jac_res_list)
    print("\n")

    print("Gauss Seidel Iterations")
    print(f"L1-Norm of Solution Vector: {get_l1_norm_vector(gauss_solution_x)}")
    print(f"L2-Norm of Solution Vector: {get_l2_norm_vector(gauss_solution_x)}")
    print(f"L-Infinity-Norm of Solution Vector: {get_linf_norm_vector(gauss_solution_x)}")
    plot_graph("Gauss Seidel Iterations", "Iterations", "L-2 Residual", gauss_iter_list, gauss_res_list)

    print("SOR Iterations")
    print(f"L1-Norm of Solution Vector: {get_l1_norm_vector(sor_solution_x)}")
    print(f"L2-Norm of Solution Vector: {get_l2_norm_vector(sor_solution_x)}")
    print(f"L-Infinity-Norm of Solution Vector: {get_linf_norm_vector(sor_solution_x)}")
    plot_graph("All Iterations", "Iterations", "L-2 Residual", sor_iter_list, sor_res_list)

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
    
    vanilla_sol = vanilla_matrix_solution(A_matrix, b_vector)
    print("Vannila Iterations")
    print(f"L1-Norm of Solution Vector: {get_l1_norm_vector(vanilla_sol)}")
    print(f"L2-Norm of Solution Vector: {get_l2_norm_vector(vanilla_sol)}")
    print(f"L-Infinity-Norm of Solution Vector: {get_linf_norm_vector(vanilla_sol)}")
