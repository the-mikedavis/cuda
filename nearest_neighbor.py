import numpy as np

def nearest_neighbor(dataset):
    a, b = dataset
    nn = np.zeros((a.size/3,))
    for i in range(0, a.size/3):
        # returns the index of the minimum element in the array
        dists = [euclidean_dist(xyz(a, i), squash(b_j)) for b_j in b.T]

        index = 0
        min_elem = float('inf')
        for j in range(0, len(dists)):
            if dists[j] < min_elem:
                min_elem = dists[j]
                index = j
        nn[i] = index
    return nn

def xyz(matrix, entry):
    return squash(matrix[:, entry])

def squash(matrix):
    return np.squeeze(np.asarray(matrix))

def euclidean_dist(p_a, p_b):
    return np.sqrt((p_b[0] - p_a[0]) ** 2 + (p_b[1] - p_a[1]) ** 2)
