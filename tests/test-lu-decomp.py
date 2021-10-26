import src.utils as u
import src.lu_decomp as lu
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

    vanilla_x_solution = u.vanilla_matrix_solution(A, B)
    lu_x_solution = lu.lu_matrix_solution(A, B)

    vanilla_row, vanilla_col = u.get_row_col_matrix(vanilla_x_solution)
    print(
        f"The solution vector using the vanilla approach of Inverse(A) * B = X with dimensions {vanilla_row}x{vanilla_col} using numpy library:"
    )
    u.print_matrix(vanilla_x_solution)

    print("\n")

    lu_row, lu_col = u.get_row_col_matrix(lu_x_solution)
    print(
        f"The solution vector using the LU-Decomposition, Forward Substitution and Backward Substitution to find solution vector x with dimensions {lu_row}x{lu_col} from scratch:"
    )
    u.print_matrix(lu_x_solution)

    print("\n")

    relative_error = u.calculate_l2_relative_error(vanilla_x_solution, lu_x_solution)
    print(f"The relative error between X and Xlu is {relative_error * 100}%")