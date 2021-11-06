import math
import src.interpolation.utils as utils
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


if __name__ == "__main__":
    x = [-2, -1, 0, 1, 2]
    coords = generate_functional_values(x)
    print(coords)
    poly = vanilla_polynomial_regression(coords)
    print(matrix.round_matrix(poly))