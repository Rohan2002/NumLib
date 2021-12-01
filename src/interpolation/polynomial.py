import math
import src.interpolation.utils as utils
import src.utils as vector_utils
import src.matrix as matrix

def generate_functional_values(x: list):
    y = []
    for x_elem in x:
        y_elem = (math.exp(x_elem) - math.exp(-x_elem)) / (math.exp(x_elem) + math.exp(-x_elem))
        y.append((x_elem, y_elem))
    return y

def vanilla_polynomial_regression(list_x_y: tuple):
    """
        Perform polynomial regression on a list of (x,y) values.

        list = [(x1,y1), (x2,y2), (x3,y3)]
        N = 3
        Vander_Monde = [[1, x1^1, x1^2],[1, x2^1, x2^2],[1, x3^1, x3^2]]

    """
    if len(list_x_y[0]) !=2:
        raise ValueError("List must contain tuples with only x and y values!")
    
    x_list = [coord[0] for coord in list_x_y]
    y_list = [[coord[1]] for coord in list_x_y]
    
    vander_monde = utils.get_vander_monde_matrix(x_list, N=len(x_list))
    inv_vander_monde = matrix.get_matrix_inverse(vander_monde)
    solution = matrix.multiply_matrices(inv_vander_monde, y_list)

    return solution

def get_A_and_B_and_Z(list_x_y):
    if len(list_x_y[0]) !=2:
        raise ValueError("List must contain tuples with only x and y values!")
    n = len(list_x_y)
    dx = [0] * (n-1)
    h = [0] * (n-1)
    for j in range(n-1):
        dx[j] = list_x_y[j+1][0] - list_x_y[j][0]
        h[j] = (list_x_y[j+1][1] - list_x_y[j][1]) / dx[j]
    print(list_x_y)
    print(h)
    A = []
    for row in range(n-2):
        row_vector = [0] * (n-2) # Init a sparse row
        delta_x = dx[row]
        for col in range(len(row_vector)):
            if row == col:
                row_vector[row] = 4. * delta_x
                if row > 0:
                    row_vector[row-1] = delta_x
                if row < len(row_vector) - 1:
                    row_vector[row+1] = delta_x
        A.append(row_vector)
    
    B = vector_utils.create_empty_vector(axis=1, length=n-2, fill=0)
    for i in range(len(h) - 1):
        B[i] = [6. * (h[i+1] - h[i])]
    
    inv = matrix.get_matrix_inverse(A)
    Z = matrix.multiply_matrices(inv, B)
    Z = [[0]] + Z + [[0]]
    return A, B, Z

def S(i, x_input, x, y, z):
    delta_x = 1
    cubic_term = z[i+1][0]/(6*delta_x)*((x_input-x[i])**3) - z[i][0]/(6.*delta_x)*((x_input-x[i+1])**3)
    other_term = (y[i+1]/delta_x-delta_x/6*z[i+1][0])*(x_input-x[i])-(y[i]/delta_x-delta_x/6*z[i][0])*(x_input-x[i+1])
    return cubic_term + other_term

if __name__ == "__main__":
    x = [-2, -1, 0, 1, 2]
    coords = generate_functional_values(x)
    x_list = [coord[0] for coord in coords]
    y_list = [coord[1] for coord in coords]
    # print(coords)
    # poly = vanilla_polynomial_regression(coords)
    # print(matrix.round_matrix(poly))
    A, B, Z = get_A_and_B_and_Z(coords)
    print(A, B, Z)
    x_input = 1.5
    for i in range(4):
        s = S(i, x_input, x_list, y_list, Z)
        print(s)