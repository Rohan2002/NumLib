import math
import matplotlib.pyplot as plt


def f(t, u):
    return (10 * math.cos(t / 10)) - u


def exact_solution(t, u_init, lamb=1):
    return (
        (u_init - ((1000 * lamb) / (1 + (100 * (lamb ** 2))))) * math.exp(-1 * lamb * t)
    ) + (100 * ((10 * lamb * math.cos(t / 10)) + math.sin(t / 10))) / (
        1 + (100 * (lamb ** 2))
    )


def euler_forward(u, t, dt, func):
    return u + dt * func(t, u)


#
def rk4(u, t, dt, func):
    K1 = dt * func(t, u)
    K2 = dt * func(t + dt / 2.0, u + K1 / 2.0)
    K3 = dt * func(t + dt / 2.0, u + K2 / 2.0)
    K4 = dt * func(t + dt, u + K3)
    one_over_six = 1.0 / 6.0
    return u + one_over_six * (K1 + 2 * K2 + 2 * K3 + K4)


def error(n, t):
    return abs((n - t) / t)


def run():
    times = []  # store t0, t1, t2, ....
    exact_values = []
    euler_values = []
    rk4_values = []

    t = 0
    times.append(t)
    exact_val = 0
    exact_values.append(exact_val)
    euler_val = 100
    euler_values.append(euler_val)
    rk4_val = 0
    rk4_values.append(rk4_val)

    dt = 0.01
    # 2nd bug: we don't need initial value: already known.
    # iteration starts from 1 !! Otherwise cur_time = t + dt * 0 = t = 0
    for i in range(1, 4):
        # 1st bug: t is actually t0, so cur_time should be t0 + i * dt
        cur_time = t + dt * i
        times.append(cur_time)

        # exact value of u
        exact_val = exact_solution(cur_time, 100)
        exact_values.append(exact_val)

        # euler forward approximation
        euler_val = euler_forward(u=euler_val, t=cur_time - dt, dt=dt, func=f)
        euler_values.append(euler_val)
        print(
            f"True Value: {exact_val}, Euler Value: {euler_val}, error: {error(euler_val, exact_val)},  prev_time = {cur_time - dt} curr_time = {cur_time}, f(u, t) = {f(cur_time - dt, euler_val)}\n"
        )

        # rk4 approximation
        rk4_val = rk4(u=rk4_val, t=cur_time - dt, dt=dt, func=f)
        rk4_values.append(rk4_val)

    plt.plot(times, exact_values, label="exact solution")
    plt.plot(times, euler_values, label="euler forward")
    # plt.plot(times, rk4_values, label="rk4")
    plt.legend()
    plt.savefig("tmp.png")


run()
