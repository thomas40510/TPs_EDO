import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from solvers import *

g = 9.81
l = 1


def pendulus(t, Theta):
    return np.array([Theta[1], -g / l * np.sin(Theta[0])])


if __name__ == '__main__':
    print("TP2")
    t0 = 0
    tf = 10
    y0 = np.array([np.pi / 2, 0])
    n = 1000
    t, y = RK4(pendulus, t0, tf, y0, n)
    plt.plot(t, y[:, 0], label="RK4")
    t, y = RK2(pendulus, t0, tf, y0, n)
    plt.plot(t, y[:, 0], label="RK2")
    plt.legend()
    plt.show()
