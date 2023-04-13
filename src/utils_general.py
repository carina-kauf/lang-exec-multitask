import torch
import numpy as np
from scipy.spatial.distance import cdist
import random
import re

def set_seed(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if not cuda:
            raise ValueError("WARNING: You have a CUDA device, so you should probably run with --cuda")


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def find_matches(d, item):
    """ Find matches for item in dictionary keys
    Used in function: get_dataset_names
    Args:
        d: dictionary
        item: item to be matched against keys

    Returns:
        dataset value for matched regex
    """
    for k in d:
        if re.match(k, item):
            return d[k]


def mask2d(N_x, N_y, cutoff, periodic):
    """Define anatomical mask for h2h weights in CTRNN
    Source: https://github.com/mikailkhona/Multi-Task-learning/blob/main/Multitask_with_mask.ipynb
    Create a 2d mask of distances between neurons
    A neuron is connected to all neurons within a certain distance
    Args:
        :param N_x: number of neurons in x direction
        :param N_y: number of neurons in y direction
        :param cutoff: cutoff distance
        :param periodic: if True, use periodic boundary conditions
    Returns:
        :2d mask

    # cutoff = d in paper
    # apply anatomical mask on h2h weights
    # model.rnn.h2h.weight.data = model.rnn.h2h.weight.data*(mask2d)
    """

    # create 2d sheet of coordinates
    x1 = np.linspace(-N_x // 2, N_x // 2 - 1, N_x)
    x1 = np.expand_dims(x1, axis=1)
    x2 = np.linspace(-N_y // 2, N_y // 2 - 1, N_y)
    x2 = np.expand_dims(x2, axis=1)
    x_coordinates = np.expand_dims(np.repeat(x1, N_y, axis=0).reshape(N_x, N_y).transpose().flatten(), axis=1)
    y_coordinates = np.expand_dims(np.repeat(x2, N_x, axis=0).reshape(N_x, N_y).flatten(), axis=1)

    # calculate torus distance on 2d sheet
    distances_x = cdist(x_coordinates, x_coordinates)
    distances_y = cdist(y_coordinates, y_coordinates)

    if periodic:
        distances_y = np.minimum(N_y - distances_y, distances_y)
        distances_x = np.minimum(N_x - distances_x, distances_x)

    distances = np.sqrt(np.square(distances_x) + np.square(distances_y))
    dist = distances.reshape(N_y, N_x, N_y, N_x)
    dist = dist.reshape(N_x * N_y, N_x * N_y)
    assert dist.shape == (N_x * N_y, N_x * N_y) #added by me
    # mask connections based on distance
    dist[dist < cutoff] = 1
    dist[dist > cutoff - 1] = 0
    return dist


def mask2d_3(hidden_dim, cutoff, periodic):
    """Define anatomical mask for h2h weights in CTRNN
    Source: https://github.com/mikailkhona/Multi-Task-learning/blob/main/Multitask_with_mask.ipynb
    Create a 2d mask of distances between neurons
    A neuron is connected to all neurons within a certain distance
    Args:
        :param hidden_dim: number of neurons in the 1D direction
        :param cutoff: cutoff distance
        :param periodic: if True, use periodic boundary conditions
    Returns:
        :2d mask

    # cutoff = d in paper
    # apply anatomical mask on h2h weights
    # model.rnn.h2h.weight.data = model.rnn.h2h.weight.data*(mask2d)
    """

    # create 1D sheet of coordinates
    x = np.linspace(-hidden_dim // 2, hidden_dim // 2 - 1, hidden_dim)

    # create a meshgrid of 1D coordinates to get the 2D sheet of coordinates
    X1, X2 = np.meshgrid(x, x)
    X1, X2 = X1.flatten(), X2.flatten()

    # calculate torus distance on 2D sheet
    distances_x = cdist(X1.reshape(-1, 1), X1.reshape(-1, 1))
    distances_y = cdist(X2.reshape(-1, 1), X2.reshape(-1, 1))

    if periodic:
        distances_y = np.minimum(hidden_dim - distances_y, distances_y)
        distances_x = np.minimum(hidden_dim - distances_x, distances_x)

    distances = np.sqrt(np.square(distances_x) + np.square(distances_y))
    dist = distances.reshape(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
    dist = dist.reshape(hidden_dim**2, hidden_dim**2)
    assert dist.shape == (hidden_dim**2, hidden_dim**2) #added by me
    # mask connections based on distance
    dist[dist < cutoff] = 1
    dist[dist > cutoff - 1] = 0
    return dist



def mask2d_hidden_size_2(hidden_dim, cutoff, periodic):
    """Define anatomical mask for h2h weights in CTRNN
    Source: https://github.com/mikailkhona/Multi-Task-learning/blob/main/Multitask_with_mask.ipynb
    Create a 2d mask of distances between neurons
    A neuron is connected to all neurons within a certain distance
    Args:
        :param hidden_dim: number of neurons in the network
        :param cutoff: cutoff distance
        :param periodic: if True, use periodic boundary conditions
    Returns:
        :2d mask of distances of shape (hidden_dim, hidden_dim)
    """
    # round up square root of hidden_dim to nearest integer
    N = int(np.ceil(np.sqrt(hidden_dim)))

    # create 2d sheet of coordinates
    x = np.arange(N) - N // 2
    xx, yy = np.meshgrid(x, x)
    x_coordinates = xx.reshape(-1, 1)
    y_coordinates = yy.reshape(-1, 1)

    # calculate torus distance on 2d sheet
    distances_x = cdist(x_coordinates, x_coordinates)
    distances_y = cdist(y_coordinates, y_coordinates)

    if periodic:
        distances_y = np.minimum(N - distances_y, distances_y)
        distances_x = np.minimum(N - distances_x, distances_x)

    distances = np.sqrt(np.square(distances_x) + np.square(distances_y))
    dist = distances.reshape(N * N, N * N)
    assert dist.shape == (hidden_dim, hidden_dim)

    # mask connections based on distance
    dist[dist < cutoff] = 1
    dist[dist > cutoff - 1] = 0
    return dist



def mask2d_hidden_size(hidden_dim, cutoff, periodic):
    """Define anatomical mask for h2h weights in CTRNN
    Source: https://github.com/mikailkhona/Multi-Task-learning/blob/main/Multitask_with_mask.ipynb
    Create a 2d mask of distances between neurons
    A neuron is connected to all neurons within a certain distance
    Args:
        :param hidden_dim: number of neurons in the network
        :param cutoff: cutoff distance
        :param periodic: if True, use periodic boundary conditions
    Returns:
        :2d mask of distances of shape (hidden_dim, hidden_dim)
    """

    # create 2d sheet of coordinates
    N = int(np.sqrt(hidden_dim))
    x1 = np.linspace(-N // 2, N // 2 - 1, N)
    x1 = np.expand_dims(x1, axis=1)
    x2 = np.linspace(-N // 2, N // 2 - 1, N)
    x2 = np.expand_dims(x2, axis=1)
    x_coordinates = np.expand_dims(np.repeat(x1, N, axis=0).reshape(N, N).transpose().flatten(), axis=1)
    y_coordinates = np.expand_dims(np.repeat(x2, N, axis=0).reshape(N, N).flatten(), axis=1)

    # calculate torus distance on 2d sheet
    distances_x = cdist(x_coordinates, x_coordinates)
    distances_y = cdist(y_coordinates, y_coordinates)

    if periodic:
        distances_y = np.minimum(N - distances_y, distances_y)
        distances_x = np.minimum(N - distances_x, distances_x)

    distances = np.sqrt(np.square(distances_x) + np.square(distances_y))
    dist = distances.reshape(N, N, N, N)
    dist = dist.reshape(hidden_dim, hidden_dim)
    assert dist.shape == (hidden_dim, hidden_dim)

    # mask connections based on distance
    dist[dist < cutoff] = 1
    dist[dist > cutoff - 1] = 0
    return dist


def sparsemask2d(N_x, N_y, sparsity):
    """
    Define Erdos-Renyi sparse mask for h2h weights in CTRNN
    Source: https://github.com/mikailkhona/Multi-Task-learning/blob/main/Multitask_with_mask.ipynb
    Args:
        :param N_x: number of neurons in x direction
        :param N_y: number of neurons in y direction
        :param sparsity: sparsity of the mask
    Returns:
        :2d mask

    # sparsity for d=2 is 0.03228759765625
    # sparsity for d=3 is 0.0836
    # sparsity for d=4 is 0.1423
    """
    elements = np.random.uniform(0, 1, (N_x, N_y))
    mask = (elements < sparsity).astype(int)
    return mask

if __name__ == "__main__":
    # test mask2d function
    hidden_size = 300
    import matplotlib.pyplot as plt
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # test
    h2h_mask2d = mask2d_3(hidden_dim=hidden_size, cutoff=3, periodic=False)
    plt.imshow(h2h_mask2d, cmap="jet")
    plt.colorbar()
    plt.title("2d Mask")
    plt.show()