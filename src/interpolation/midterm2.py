"""
    My midterm 2 code that includes Cubic spline, and trapezoid rule.
"""

import numpy as np
import matplotlib.pyplot as plt

def in_range(a, amin, amax):
    return a >= amin and a <= amax


def construct_a(number_of_data_points, delta_x):
    A_size = number_of_data_points - 2
    A_mat = np.zeros(shape=(A_size, A_size))

    for i in range(A_size):
        A_mat[i][i] = 2.0 * (2 * delta_x)
        if in_range(i - 1, 0, A_size - 1):
            A_mat[i][i - 1] = delta_x
        if in_range(i + 1, 0, A_size - 1):
            A_mat[i][i + 1] = delta_x
    return A_mat


def H(i, y, delta_x):
    return (y[i + 1] - y[i]) / delta_x


def construct_b(a, y, delta_x):
    B = np.zeros(shape=(len(a), 1))
    for i in range(len(a)):
        B[i] = 6.0 * (H(i + 1, y, delta_x) - H(i, y, delta_x))
    return B


def construct_z(a, b, n_dp):
    z = np.zeros(shape=(len(a), 1))
    z = np.matmul(np.linalg.inv(a), b)

    z_tmp = z.copy()

    z = np.zeros(shape=(n_dp, 1))
    for i in range(1, n_dp - 1):
        z[i] = z_tmp[i - 1]

    z[0] = 0
    z[n_dp - 1] = 0
    return z


def S(i, x_input, x, y, z, delta_x):
    cubic_term = z[i + 1] / (6 * delta_x) * ((x_input - x[i]) ** 3) - z[i] / (
        6.0 * delta_x
    ) * ((x_input - x[i + 1]) ** 3)
    other_term = (y[i + 1] / delta_x - delta_x / 6 * z[i + 1]) * (x_input - x[i]) - (
        y[i] / delta_x - delta_x / 6 * z[i]
    ) * (x_input - x[i + 1])
    return cubic_term + other_term


def cubic_interpolate(x, y, delta_x, n_dp, points_to_interpolate_on):
    a = construct_a(n_dp, delta_x)
    b = construct_b(a, y, delta_x)
    z = construct_z(a, b, n_dp)

    results = np.zeros(shape=(n_dp, 1))
    for i in range(n_dp - 1):
        # Interpolate for the specific x_input
        results[i] = S(i, points_to_interpolate_on[i], x, y, z, delta_x)
    results[n_dp - 1] = S(n_dp - 2, x[n_dp - 1], x, y, z, delta_x)
    return results


def trapezoid_rule(x, y):
    """
        Question 1 part 3. trapezoidal sum 
    """
    delta_x = (x[len(x) - 1] - x[0]) / (len(x) - 1)
    sum = 0
    for i in range(1, len(x) - 1):
        sum += y[i]
    return ((2 * sum) + y[0] + y[len(x) - 1]) * delta_x / 2


if __name__ == "__main__":
    # September 2021
    time = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    cases = [
        393.71,
        396.14,
        387,
        400.57,
        419,
        420.86,
        449.29,
        473.86,
        499.43,
        514.86,
        473.43,
        579.43,
    ]

    delta_x = 2  # For now assume we choose datapoints with delta x as constant.
    n_dp = 12

    days_to_interpolate_on = time
    interpolate = cubic_interpolate(time, cases, delta_x, n_dp, days_to_interpolate_on)

    plt.scatter(time, cases)
    plt.xlabel("X: Days in September 2021")
    plt.ylabel("Y: 7 day moving average")
    plt.plot(days_to_interpolate_on, interpolate)
    plt.savefig("covid_nj_cubic_spline_rohan_deshpande.png")

    num_of_ppl = trapezoid_rule(time, cases)
    print(
        f"The number of people infected with covid from september 1 to 23 2021 is {num_of_ppl}."
    )
