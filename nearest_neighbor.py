import numpy as np

def nearest_neighbor(dataset):
    a, b = dataset
    nn = np.zeros((a.size/3,))
    for i in range(0, a.size/3):
        dists = np.array([euclidean_dist(np.squeeze(np.asarray(a[:, i])), xyz) for xyz in np.squeeze(np.asarray(b))])
        # returns the index of the minimum element in the array
        nn[i] = np.argmin(dists)
    return nn

def euclidean_dist(p_a, p_b):
    return np.sqrt((p_b[0] - p_a[0]) ** 2 + (p_b[1] - p_a[1]) ** 2)
