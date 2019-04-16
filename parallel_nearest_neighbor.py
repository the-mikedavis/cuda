import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void euclidean_dist(float *dest, float *a, float *b) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = 2 * idx;
  float dist;

  // arrays are flattened when copied, so (a[0][0], a[0][1]) becomes (a[0], a[1])
  printf("%d: i=%d a=(%f, %f) b=(%f, %f)\\n", idx, i, a[i], a[i + 1], b[i], b[i + 1]);

  dist = hypotf(b[i] - a[0], b[i + 1] - a[1]);

  printf("dist(a, b)=%f\\n", dist);

  dest[idx] = dist;
}
""")

parallel_euclidean_dist = mod.get_function("euclidean_dist")

def nearest_neighbor(dataset):
    a, b = dataset
    nn = np.zeros((a.size/3,))
    for i in range(0, a.size/3):
        # returns the index of the minimum element in the array
        bs_xy = np.delete(b.T, 2, 1)
        dists = euclidean_dists(xyz(a, i), bs_xy)

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

# for posterity
def euclidean_dist(p_a, p_b):
    return np.sqrt((p_b[0] - p_a[0]) ** 2 + (p_b[1] - p_a[1]) ** 2)

# (x, y) of a, [(x, y)] is bs, number of points
def euclidean_dists(a, bs, n):
    # number of points
    n = b.size / len(bs)
    print "n"
    print n

    # bzero the dest array
    dest = numpy.zeros((n,)).astype(numpy.float32)

    # elements a[0], a[1]
    a_point = a[0:2]

    parallel_euclidean_dist(drv.Out(dest), drv.In(a[0:2]), drv.In(bs), block=(n, 1, 1), grid=(1, 1))

    return dest
