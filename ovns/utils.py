from time import process_time
from datetime import timedelta as td
from ovns.initialize import *
import numpy as np
from numba import *
from numba import njit
import numba

def __to_len_classes(ss):
    n_ss = {}
    for i,s in enumerate(ss):
        n_s = len(s)
        if n_s not in n_ss:
            n_ss[n_s] = []
        n_ss[n_s].append(i)
    return n_ss

@njit
def __svns_score(H_w, Ho_w, H, Ho, k):
    return (H_w / Ho_w) + (k - (H & Ho).sum()) / k

@njit
def __update_degree_vecs(A, alpha, beta, xi, xj, inplace=False):
    alpha_p = alpha if not inplace else alpha.copy()
    beta_p = beta if not inplace else beta.copy()
    for y in range(A.shape[0]):
        alpha_p[y] = alpha[y] - A[y,xi] + A[y,xj]
        beta_p[y] = beta[y] + A[y,xi] - A[y,xj]
    return alpha_p, beta_p

@njit
def __create_bvns_array(A):
    """Compute neighbor array for bvns such that ith row corresponds to node i and
    indeces of nodes adjacent to i are the first elements in the row, while end of
    the rows are padded with -1.
    """
    n = A.shape[0]

    Ap = np.zeros((n,n), dtype=np.int32) - 1
    for i in range(n):
        nz = np.where(A[i,:])[0]
        n_nz = nz.shape[0]
        Ap[i,:n_nz] = nz
        Ap[i,n_nz:] = -1
    return Ap

@njit
def __create_beam_array(A, A_as, w_thres):
    """Compute a beam array out of adjacency matrix A. In a beam array each row
    i will contain the indexes of all connected nodes for node i in sorted order
    based on the link weight."""

    n = A.shape[0]

    A_beam = np.zeros((n,n), dtype=np.int32) - 1
    maxlens = np.zeros(n, dtype=np.int32) + n
    for i in range(n):
        j = 0
        for k in A_as[i,:]:
            if A[i,k] >= w_thres:
                A_beam[i,j] = k
                j+=1
            else:
                if j < maxlens[i]:
                    maxlens[i] = j
                break

    return A_beam[:,:maxlens.max()], maxlens.mean()

@njit
def __create_beam_array_constant_width(A, A_as, w_thres):
    """Compute a beam array out of adjacency matrix A. In a beam array each row
    i will contain the indexes of all connected nodes for node i in sorted order
    based on the link weight."""

    #print('Beam width set')
    n_beam = 6
    n = A.shape[0]

    A_beam = np.zeros((n,n_beam), dtype=np.int32) - 1
    maxlen = n
    for i in range(n):
        for j in range(n_beam):
            k = A_as[i,j]
            if A[i,k] > 0.0:
                A_beam[i,j] = k
                j+=1
            else:
                if j < maxlen:
                    maxlen = j
                break

    if maxlen < n_beam:
        A_beam = A_beam[:,:maxlen]

    return A_beam

def to_numpy_array(U: list) -> np.array:
    """
    Tansforms list U of lists u \in U, where each u represents a list of node
    ids as integers, to numpy 2d array of size n x p where n is the length of
    the list U and p is the length of the longest item u in the list. Note
    that for shorter elements u s.t. len(u) < p, the rows are padded from the
    right with -1.
    """
    n, k = len(U), len(sorted(U, key=len)[-1])
    Uc = np.zeros((n,k), dtype=np.int64) - 1
    for i,u in enumerate(U):
        Uc[i,:len(u)] = u

    return Uc

@njit
def mean_ndiag(B) -> int:
    n = B.shape[0]
    return (B.sum()-np.diag(B).sum())/(n*(n-1))

flatten = lambda x : list(chain.from_iterable(x))
overlap_coefficient = lambda A,B: len(A & B) / np.min([len(A),len(B)])

@njit
def unique_nodes(U: np.array, n: int) -> np.array:
    """Function to find unique nodes in U.

    Notes
    -----
    This implementation is ca. 1-2 orders of magnitude faster
    than np.unique(U) or set(U.ravel()), but requires knowledge
    about the number of nodes in the network.
    """
    set_u = np.array([False]*n)
    for u in U:
        for v in u:
            if v != -1:
                set_u[v] = True
    return np.where(set_u)[0]

@njit
def sub_sum(A, u):
    """Function for computing subgraph/graphlet weight."""
    eps = 0
    for ui in u:
        for uj in u:
            eps += A[ui,uj]
    return eps / 2

@njit
def choice(inp_array, size, replace=False, p=None):
    """
    Sample `size` elements from `inp_array` with or without replacement.

    Parameters:
        inp_array (ndarray): The input array to sample from.
        size (int): The number of elements to sample.
        replace (bool, optional): Whether to sample with replacement (default: False).
        p (ndarray, optional): The probability weights for each element in `inp_array`.
            If not provided, a uniform distribution is assumed.

    Returns:
        ndarray: An array of shape (`size`,) with the sampled elements.
    """
    n_max_resample = size*10

    n = inp_array.shape[0]

    if p is None:
        p = np.ones(n, dtype=np.float64) / n
    if p.shape[0] != n:
        raise ValueError("Weight vector and input array dimensions are not equal")
    if n == 0:
        raise ValueError("Cannot take a sample with size 0")
    elif size > n:
        raise ValueError("Cannot take a larger sample than population when " \
                         "sampling with replacement")
    elif n == size:
        return inp_array

    wc = np.cumsum(p)
    m = wc[-1]

    sample = np.empty(size, inp_array.dtype)
    sample_idx = np.full(size, -1, np.int32)

    i = n_resample = 0
    while i < size:
        r = m * np.random.rand()
        idx = np.searchsorted(wc, r, side='right')
        if not replace:
            re_sample = False
            for j in range(i):
                if sample_idx[j] == idx:
                    re_sample = True
                    n_resample += 1
            if re_sample: continue
        n_resample = 0
        sample[i] = inp_array[idx]
        sample_idx[i] = idx
        i += 1
    assert n_resample < n_max_resample
    return sample

def compute_random_reference(A, result, n_draws=1000, seed=None):
    """
    Compute the mean value of a random reference for OVNS convergence diagnostics.

    Parameters:
        A (array-like): The input array or matrix.

        result (dict): The result dictionary obtained from OVNS.

        n_draws (int, optional): The number of random draws to perform.
            Defaults to 1000.

        seed (int or None, optional): Seed for the random number generator.
            Defaults to None.

    Returns:
        float: The mean value of the random reference.

    """
    k = result['params']['k']
    n = result['alpha'].shape[0]

    rng = np.random.default_rng(seed=seed)
    Hws = np.zeros(n_draws)

    for i in range(n_draws):
        idxs = rng.choice(n, k, replace=False)
        Hw = sub_sum(A, idxs)
        Hws[i] = Hw

    return Hws.mean()

try:
    import matplotlib.pyplot as plt

    def plot_convergence(res, A=None, save=None,
                     title='Convergence',
                     include_initialization=True,
                     double_xaxis=True,
                     ax=None,
                     relative=True,
                     **kwargs):
        """
        Generate a convergence plot to visualize OVNS convergence diagnostics.

        Parameters:
            res (dict): Output from OVNS.
                - 'run_trace': A list of tuples (Y, X) representing the convergence trace.
                - 'iterations': Total number of iterations.
                - 'running_time': Total running time in seconds.
            A (np.array of shape (n,n), optional): OVNS input network as adjacency matrix.
                                                Defaults to None.
            save (str, optional): The path to save the plot image file. Defaults to None.

            title (str, optional): The title of the plot. Defaults to 'Convergence'.

            include_initialization (bool, optional): Whether to include the initialization
                point in the plot. If True, then A is expected to be passed for computing the
            random initial solution as reference. Defaults to True.

            double_xaxis (bool, optional): Whether to include a secondary x-axis to display
                running time in hours. Defaults to True.

            ax (matplotlib.axes.Axes, optional): The matplotlib axes to plot on.
                If not provided, a new figure and axes will be created. Defaults to None.

            relative (bool, optional): Whether to plot the fraction of relative improvement
                or the objective function value. Defaults to True.

            **kwargs: Additional keyword arguments to be passed to the plot function calls.

        Returns:
            None (displays the plot or saves it to a file).
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

        ax.set_title(title, y=1.15)

        if include_initialization:
            assert type(E_input) is np.ndarray, 'Pass adjacency matrix A as numpy array'
            Y0 = compute_random_reference(A, res)

        Y, X = zip(*res['run_trace'])
        X = list(X) + [res['iterations']]

        ymin = np.min(Y)
        ymax = np.max(Y)
        if include_initialization:
            Y_init = np.array([Y0] + list(Y))
            Y_init = (Y_init - Y0) / (ymax - Y0) if relative else Y_init
            ax.plot(np.array(X), Y_init, '.-.', alpha=1.0, label='including $H_0$', **kwargs)
            ax.plot([X[0],X[-1]],[0]*2 if relative else [Y0]*2,'-.',alpha=.5, lw=0.5,
                color='#111', label='random baseline')

        Y_opt = np.array([np.nan] + list(Y))
        Y_opt = (Y_opt - ymin) / (ymax - ymin) if relative else Y_opt
        ax.plot(np.array(X), Y_opt, '.-', alpha=1.0, label='excluding $H_0$', **kwargs)

        ax.set_xlabel('Number of iterations')
        ylabel = 'Fraction of relative improvement $I$' if relative else 'Objective function value'
        ax.set_ylabel(ylabel)

        x_ticks = ax.get_xticks()
        xmin, xmax = ax.get_xlim()

        if double_xaxis:
            iter_rate = res['iterations'] / res['running_time']
            iter2time = lambda x: x*iter_rate**-1 / 3600
            time2iter = lambda x: 3600*x*iter_rate
            ax2 = ax.secondary_xaxis('top', functions=(iter2time, time2iter))
            ax2.set_xlabel('Running Time (hours)')

        ax.legend()
        plt.tight_layout()
        if save:
            plt.savefig(save)

except ImportError:
    pass