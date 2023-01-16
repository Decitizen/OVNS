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