import math
import sys
import matplotlib.pyplot as plt


def func(x):
    """
    Create a function to evaluate the roots of it using bisection method
    """
    return math.exp(math.cos(x) + math.cos(x ** 2)) + math.cos(x) - 1


def bisection(f, a, b, max_iters, eps=1e-6):
    """
        Author: Rohan Deshpande.
        This program will find the root c such that f(x) = 0 using bisection method.
    """
    if f(a) * f(b) >= eps:
        # Check wether there exists c in the domain [a, b] (b>a) or [b, a] (b<a) such that will be f(c) = 0
        print("Error: Could not use bisection method")
        sys.exit(0)
    
    # Variables for bisection data
    iteration = 0
    steps = []
    roots = []

    # Go for MAX_ITERATIONS ONLY
    while iteration <= max_iters:
        
        # Calculate initial bisection root
        c = 0.5 * (a + b)
        steps.append(iteration)
        roots.append(abs(f(c)))
        print(
            f"Iterations: {iteration}, a: {round(f(a),4)}, f(a): {round(f(a),4)}, b: {round(b,4)}, f(b): {round(f(b),4)}, c: {round(c,4)}, f(c): {round(f(c),4)}"
        )

        # Incremement Iteration
        iteration += 1

        # The root c will be valid if its 0 or less than epsilon
        if f(c) == 0 or abs(f(c)) < eps:
            break
        
        # The bisection substitution happens here based on the signs of f(a) and f(c).
        if (f(a) < 0 and f(c) < 0) or (f(a) > 0 and f(c) > 0):
            a = c
        else:
            b = c
    
    # Finally, print the plot the data over here
    print(f"The root will be x=c={round(c,4)} and y=f(c)={round(f(c),4)}")

    plt.xlabel("k Iterations", fontsize=18)
    plt.ylabel("f(x_k) value", fontsize=18)
    plt.plot(steps, roots)
    plt.show()
    return c, f(c)


if __name__ == "__main__":
    bisection(f=func, a=0, b=2, max_iters=100, eps=1e-6)
