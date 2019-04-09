'''rigid transformation optimization
with analytical gradient
'''

import numpy as np

# assuming X and Y have the same number of points (n x 2) initially only 2d transformations


def matchingCost(X, Y, theta, b, gradient=True):
    T = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    YT = np.dot(Y, T.transpose())
    E = X - YT - b[None, :]
    cost = np.mean(np.sum(np.square(E),axis=1)) / 2 
    if gradient is True:
        grad_theta = np.mean(E[:,0] * YT[:,1] - E[:,1] * YT[:,0])
        grad_b = -np.mean(E, axis=0)
        return cost, (grad_theta, grad_b)
    else:
        return cost
# We run the updates for 
def transformPointCloud(X, theta, b):
    T = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    XT = np.dot(X, T.transpose())
    return XT + b[None, :]

    
def matchPairs(X, Y, mu=0.01, momentum=0.5, max_iter=100, tol=1e-5, init_theta=None, init_b=None):
    if init_theta is None:
        theta = 0.0
    else:
        theta = init_theta
    if init_b is None:
        b = np.zeros(shape=(2,), dtype=np.float32)
    else:
        b = init_b

    cost = np.zeros(max_iter + 1, dtype=np.float32)
    cost[0] = np.Inf
    delta_theta = 0.0
    delta_b = np.zeros_like(b)
    i = 0
    while i < max_iter:
        cost[i+1], (grad_theta, grad_b) = matchingCost(X, Y, theta, b)
        delta_theta = momentum*delta_theta - mu*grad_theta
        delta_b = momentum*delta_b - mu*grad_b
        theta += delta_theta
        b += delta_b
        if np.abs(cost[i+1] - cost[i]) < tol:
            break
        i += 1
    if i == max_iter:
        print('Max number of iterations reached before tolerance level')
    else:
        print('Converged after %d iterations'%i)
    
    return transformPointCloud(Y, theta, b), cost[1:i+1]

