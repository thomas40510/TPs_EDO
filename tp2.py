import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from solvers import *

g = 9.81
l = 1


def pendulus(t, Theta):
    return np.array([Theta[1], -g / l * np.sin(Theta[0])])


def pendulus_approx(t, Theta):
    return np.array([Theta[1], -g / l * Theta[0]])


def pendulus_approx2(t, Theta):
    return np.array([Theta[1], -g / l * Theta[0] - Theta[0] ** 3 / 6])


def compare_approxs():
    t, y = RK4(pendulus, t0, tf, y0, n)
    t, y2 = RK4(pendulus_approx, t0, tf, y0, n)
    t, y3 = RK4(pendulus_approx2, t0, tf, y0, n)

    plt.plot(t, y[:, 0], label="$\sin(\\theta)$")
    plt.plot(t, y2[:, 0], label="$\\theta$")
    plt.plot(t, y3[:, 0], label="$\\theta + \\theta^3/6$")
    plt.legend()
    plt.title("Approximations aux petits angles")
    plt.show()


if __name__ == '__main__':
    print("TP2")
    t0 = 0
    tf = 10
    y0 = np.array([np.pi / 2, 0])
    n = 1000

    compare_approxs()
    t, y = RK4(pendulus, t0, tf, y0, n)
    plt.plot(t, y[:, 0], label="RK4")
    t, y = RK2(pendulus, t0, tf, y0, n)
    plt.plot(t, y[:, 0], label="RK2")
    plt.legend()
    plt.show()
