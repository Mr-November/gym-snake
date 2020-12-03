import numpy as np
import matplotlib.pyplot as plt

# When order == 2
c_1 = np.array([-np.pi / 6.0, 0.0, 0.0])
c_2 = np.array([np.pi / 6.0, np.pi / 4.0, np.pi / 12.0])
cc1 = np.vstack((c_1, 2.0 * c_1))
cc2 = np.vstack((c_2, 2.0 * c_2))
c_min = np.vstack((cc1, cc1, cc1, cc1, cc1, cc1, cc1, c_1))
c_max = np.vstack((cc2, cc2, cc2, cc2, cc2, cc2, cc2, c_2))

# Main diagnal are all zero.
# Other elements are in range [0, 1].
w_min = np.zeros((15, 15), dtype = np.float64)
w_max = np.ones((15, 15), dtype = np.float64)

# Main diagnal are all zero.
# Total number of independent variables: n.
# All elements are in range [-2 * pi, 2 * pi]
p_min = -2.0 * np.pi * np.ones((15, 1))
p_max = 2.0 * np.pi * np.ones((15, 1))
def GetPhaseMatrix(phase_array):

    dim = phase_array.shape[0]
    p = np.zeros((dim, dim), dtype = np.float64)

    for i in range(dim):
        for j in range(dim):
            p[i, j] = phase_array[i, 0] - phase_array[j, 0]

    return p

def GetPos(n, order, x):

    y = np.zeros((n, 1), dtype = np.float64)

    for i in range(n):
        for j in range(order + 1):
            if j == 0:
                y[i] += x[i, j]
            else:
                y[i] += x[i, j] * np.sin(j * x[i, order + 1])

    return y

def RungeKutta4(ode, step, x):

    k1 = step * ode(x)
    k2 = step * ode(x + k1 / 2)
    k3 = step * ode(x + k2 / 2)
    k4 = step * ode(x + k3)
    y = x + 1.0 / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    return y

# The structure of x:
# /fourier coefficients               /phase
# 0,    1,    2,  ...  ,    order,    theta
def CPGode(n, order, nu, coefficient_mat, weight_mat, phase_mat, x):

    y = np.zeros(x.shape)

    for i in range(order + 1):
        y[:, i] = 1.0 * (coefficient_mat[:, i] - x[:, i])

    for i in range(n):
        delta = 0.0
        for j in range(n):
            delta += weight_mat[i, j] * np.arctan(x[i, order + 1] - x[j, order + 1] - phase_mat[i, j])
        y[i, order + 1] = delta + np.pi * nu[i]

    return y
