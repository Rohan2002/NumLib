def get_vander_monde_matrix(x: list, N=None):
    """
        A vandermonde matrix is a mxn matrix for a list of x values [x_1,x_2,x_3,....,x_m] generated as the following
        [   [1, (x_1)^2, (x_1)^3...(x_1)^(n-1)],
            [1, (x_2)^2, (x_2)^3...(x_2)^(n-1)],
            .
            .
            .
            [1, (x_m)^2, (x_m)^3...(x_m)^(n-1)]
        ]
    """
    columns = N if N else len(x)
    mat = []
    for value in x:
        row_vector = []
        row_vector.append(1)
        for power in range(1, columns):
            row_vector.append(value ** power)
        
        mat.append(row_vector)
    return mat
