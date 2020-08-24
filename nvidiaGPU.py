from numba import njit, prange
import numpy as np

@njit(parallel = True, fastmath = True)
def E(kernel, image, f):

    E = np.zeros(image.shape, dtype = np.float32)
    for i in prange(image.shape[0]):
        for j in prange(image.shape[1]):
            newimage = f - image[i,j]
            newimage = np.abs(newimage)
            E[i,j] = np.sum(newimage*kernel)

    return E
