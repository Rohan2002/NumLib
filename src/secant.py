import math


def func(x):
    return x + math.exp(x)


def secant(
    max_iter,
    x_0,
    x_1,
    eps,
    f,
):

    for iter in range(1, max_iter + 1):
        x_guess = x_1 - (((x_1 - x_0) / (f(x_1) - f(x_0))) * f(x_1))
        error = abs(x_guess - x_1)

        print(f"Iteration = {iter}, x = {round(x_guess, 3)}, error = {round(error,3)}")
        if error == 0 or error < eps:
            print(f"Final root value is x = {x_guess} with a error of {error}")
            break
        x_0 = x_1
        x_1 = x_guess
        iter += 1


if __name__ == "__main__":
    secant(max_iter=100, x_0=-1, x_1=-1.1, eps=0, f=func)
