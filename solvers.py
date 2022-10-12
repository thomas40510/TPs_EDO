import numpy as np
from scipy.optimize import fsolve


def euler_explicite(f, t0, tf, y0, n):
    h = (tf - t0) / n
    t = np.linspace(t0, tf, n + 1)
    y = np.zeros((n + 1, len(y0)))
    y[0] = y0
    for i in range(n):
        y[i + 1] = y[i] + h * f(t[i], y[i])
    return t, y


def RK2(f, t0, tf, y0, n):
    h = (tf - t0) / n
    t = np.linspace(t0, tf, n + 1)
    y = np.zeros((n + 1, len(y0)))
    y[0] = y0
    for i in range(n):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h, y[i] + h * k1)
        y[i + 1] = y[i] + h / 2 * (k1 + k2)
    return t, y


def euler_implicite(f, t0, tf, y0, n):
    h = (tf - t0) / n
    t = np.linspace(t0, tf, n + 1)
    y = np.zeros((n + 1, len(y0)))
    y[0] = y0
    for i in range(n):
        y[i + 1] = fsolve(lambda u: u - y[i] - h * f(t[i + 1], u), y[i])
    return t, y


K = [1 / 4, (3 - 2 * np.sqrt(3)) / 12, (3 + 2 * np.sqrt(3)) / 12, 1 / 4]


def RK4(f, t0, tf, y0, n, Ks=None):
    if Ks is None:
        Ks = K
    h = (tf - t0) / n
    t = np.linspace(t0, tf, n + 1)
    y = np.zeros((n + 1, len(y0)))
    y[0] = y0
    for i in range(n):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h * Ks[1], y[i] + h * Ks[1] * k1)
        k3 = f(t[i] + h * Ks[2], y[i] + h * Ks[2] * k2)
        k4 = f(t[i] + h * Ks[3], y[i] + h * Ks[3] * k3)
        y[i + 1] = y[i] + h * (Ks[0] * k1 + Ks[1] * k2 + Ks[2] * k3 + Ks[3] * k4)
    return t, y
