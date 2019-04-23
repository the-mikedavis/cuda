import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import  pycuda.gpuarray as parray

mod = SourceModule("""
__global__ void euclidean_dist(double *dest, double *a, double *b) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = 2 * idx;
  double dist;

  // arrays are flattened when copied, so (a[0][0], a[0][1]) becomes (a[0], a[1])
  // printf("%d: i=%d a=(%f, %f) b=(%f, %f)\\n", idx, i, a[0], a[1], b[i], b[i + 1]);

  dist = hypotf(b[i] - a[0], b[i + 1] - a[1]);

  // printf("dist(a, b)=%f\\n", dist);

  dest[idx] = dist;
}

__global__ void nn(int *dest, double *a, double *b, int n) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  // arrays are flattened when copied, so (a[0][0], a[0][1]) becomes (a[0], a[1])
  const int i = 2 * idx;
  double min_value = 1.0 / 0.0;
  double dist;
  int min_index = 0;
  int j;

  for (j = 0; j < 2 * n; j += 2) {
    dist = hypotf(b[j] - a[i], b[j + 1] - a[i + 1]);
    if (dist < min_value) {
      min_value = dist;
      min_index = j/2;
    }
  }

  dest[idx] = min_index;
}
""")

parallel_nn = mod.get_function("nn")
parallel_euclidean_dist = mod.get_function("euclidean_dist")

def nearest_neighbor(dataset):
    a, b = dataset
    n = b.size/len(b)
    as_xy = np.delete(a, 2, 0)
    bs_xy = np.delete(b, 2, 0)

    dest = np.zeros((n,)).astype(np.int32)

    parallel_nn(drv.Out(dest), drv.In(as_xy), drv.In(bs_xy), np.int32(n), block=(n, 1, 1), grid=(1, 1))

    return dest

def xyz(matrix, entry):
    return squash(matrix[:, entry])

def squash(matrix):
    return np.squeeze(np.asarray(matrix))

# for posterity
def euclidean_dist(p_a, p_b):
    return np.sqrt((p_b[0] - p_a[0]) ** 2 + (p_b[1] - p_a[1]) ** 2)

# (x, y) of a, [(x, y)] is bs, number of points
def euclidean_dists(a, bs, n):
    # bzero the dest array
    dest = np.zeros((n,)).astype(np.float64)

    # elements a[0], a[1]
    # prevents an error about ndarray continuity
    a_point = a[0:2].copy(order='C')

    parallel_euclidean_dist(drv.Out(dest), drv.In(a_point), drv.In(bs), block=(n, 1, 1), grid=(1, 1))

    return dest
