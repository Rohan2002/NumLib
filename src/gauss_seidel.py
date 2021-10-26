import src.sor as sor
def gauss_seidal_iteration(A, b, eps):
    """
        Gauss Seidal Iteration is just Successive Over Relaxation but with omega value of 1.
    """
    return sor.sor_iteration(A, b, eps=eps, omega=1)