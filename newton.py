import math

def func(x):
    return x + math.exp(x)

def func_prime(x):
    return 1 + math.exp(x)

def newton(max_iter, guess, eps, f, f_prime):
    x_k = guess

    for iter in range(1, max_iter+1):
        guess = x_k - (f(x_k) / (f_prime(x_k)))
        error = abs(guess - x_k)
        
        print(f"Iteration = {iter}, x = {round(guess, 3)}, error = {round(error,3)}")
        if error == 0 or error < eps:
            print(f"Final root value is x = {guess} with a error of {error}")
            break

        x_k = guess
        iter +=1
    
if __name__ == "__main__":
    newton(100, -1, 1e-14, func, func_prime)
