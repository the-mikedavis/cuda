#!/usr/bin/env python

"""An implementation of the Gauss-Newton algorithm."""
import numpy as np
import nearest_neighbor as nn

def makeHomogeneous(Z):
    Z = Z.reshape((2, Z.size/2))
    Z = np.vstack((Z, np.ones(Z.shape[1])))

    return Z

def transform(xn, X):
    t = np.matrix([[np.cos(xn[2]), -np.sin(xn[2]), xn[0]], [np.sin(xn[2]), np.cos(xn[2]), xn[1]], [0, 0, 1]])
    y = np.matmul(t, X)
    X = y[:-1,:]

    return X.reshape((X.size, 1))

def jacobian(xn, Y):
    e = 0.000001

    y = transform(xn, Y)
    J = np.zeros((y.size, xn.size))

    y_n = transform(np.add(xn, np.array([e,0,0])), Y)
    J[:,0] = (np.true_divide(y_n - y, e)).reshape((1, y.size))
    y_n = transform(np.add(xn, np.array([0,e,0])), Y)
    J[:,1] = (np.true_divide(y_n - y, e)).reshape((1, y.size))
    y_n = transform(np.add(xn, np.array([0,0,e])), Y)
    J[:,2] = (np.true_divide(y_n - y, e)).reshape((1, y.size))

    return J # N x 3

def residuals(xn, X, Y):
    Y = transform(xn, Y)
    Y = makeHomogeneous(Y)
    N = nn.nearest_neighbor((Y, X))
    print("NN")
    print(N)
    Y = Y[:-1,:]
    X = X[:-1,:]
    Y = Y[:,N.astype(int)]
    return X.reshape((X.size,1)) - Y.reshape((Y.size,1)) # N x 1

def solve(X, Y, x0=[-0.5,1.0,0.1], tol = 1e-6, maxits = 256):
    """Gauss-Newton algorithm for solving nonlinear least squares problems.
    """
    dx = np.ones(len(x0))   # Correction vector
    xn = np.array(x0)       # Approximation of solution

    i = 0
    while (i < maxits) and (np.absolute(np.linalg.norm(dx)/np.linalg.norm(X)) > tol):
        # correction = pinv(jacobian) . residual vector
        dx = np.dot(np.linalg.pinv(jacobian(xn, Y)), residuals(xn, X, Y))
        dx = np.squeeze(np.asarray(dx))
        xn = np.add(xn, dx)            # x_{n + 1} = x_n + dx_n
        i  += 1

    Y = transform(xn, Y)
    Y = makeHomogeneous(Y)
    return Y, xn, i

