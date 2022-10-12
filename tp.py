import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from solvers import *


def evolve(m, v, u=1):
    res = [m * (2 - .05 * v), -v * (1 - .0001 * m)]
    R = np.array(res)
    return R * u


if __name__ == '__main__':
    a = 10000
    b = 3
    h = .01
    times = np.arange(0, 50, h)

    y0 = np.array((a, b))

    V = euler_explicite(evolve, 0, 50, y0, 5000)
    W = RK2(evolve, 0, 50, y0, 5000)
    Z = RK4(evolve, 0, 50, y0, 5000)

    sol = solve_ivp(evolve, (0, 50), y0, t_eval=times, method='RK45')
    plt.plot(V[0], V[1], color='blue', label=f"euler, h={h}")
    plt.plot(W[0], W[1], color='red', label=f"RK2, h = {h}")
    plt.plot(Z[0], Z[1], color='green', label=f"RK3, h = {h}")
    plt.plot(sol.y[0, :], sol.y[1, :], color='black', label=f"scipy, h = {h}")
    plt.legend()
    plt.show()
