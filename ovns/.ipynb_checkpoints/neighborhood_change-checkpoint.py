from time import process_time
from datetime import timedelta as td
import numpy as np
from numba import *
from ovns.utils import choice
from ovns.initialize import *
from ovns.utils import __update_degree_vecs

@njit
def __replace_in_h(Ho, Ho_fs, p):
    """Replace p nodes in Ho by uniformly drawn sample from the 
    complement of Ho & Ho_fs."""
    H = Ho.copy()
    H_fs = Ho_fs.copy()
    HC = np.where((Ho | Ho_fs) == False)[0]
    xis = np.random.choice(np.where(Ho)[0], size=p, replace=False)
    xjs = np.random.choice(HC, size=p, replace=False)
    H[xis] = False
    H[xjs] = True  
    return H, H_fs, xis, xjs

def __sample_up_to_class_p(p, n_ss):
    assert p >= min(n_ss.keys()), 'p should be >= shortest length class in n_ss'
    # Pick randomly from classes of length <= p
    keys_sorted = np.array(sorted(n_ss.keys())) 
    keys_idxs = keys_sorted[keys_sorted <= p]
    lens = [len(n_ss[idx]) for idx in keys_idxs]
    idx = np.random.choice(np.sum(lens), replace=False)
    idx_t = [(i,idx - sum(lens[0:i])) for i in range(len(lens)) 
             if idx < np.sum(lens[0:i+1])][0]
    out_idx = n_ss[keys_idxs[idx_t[0]]][idx_t[1]]
    return out_idx

def __replace_in_hhfs(Ho, Ho_fs, fss, n_ss, k, n, p, verbose=False):
    """Replace p nodes in both Ho and Ho_fs with uniformly drawn sample from the 
    complement of Ho & Ho_fs."""
    H = np.zeros(n, dtype=bool)
    p_t = p
    r_count = 0
    while True:
        if r_count > 5:
            # Restart in pathological cases
            p_t = min(n_ss.keys())
            rounds = 0
        H_fs = np.zeros(n, dtype=bool)
        idxs = fss[__sample_up_to_class_p(p_t, n_ss)]
        H_fs[idxs] = True
        pp = len(idxs) - (H_fs & (Ho | Ho_fs)).sum()
        if verbose: print('Deterministic update -- ', p_t - pp + len(idxs), p_t, \
                          ' --> size of new fs:', len(idxs))
        if p_t - pp + len(idxs) <= k:
            if verbose: print('Breaking out -- ', p_t - pp + len(idxs), p_t)
            break
        else:
            r_count += 1
            p_upd = p + np.random.randint(0, r_count)
            p_t = p_upd if p_upd <= k else p 
            if verbose: print('Random -- ', p_t)
            
    p = p_t 
    if verbose: print(f':: Picked new fs node configuration of length {len(idxs)}')
    if verbose: print(f':: Number of unique new nodes {pp}')
    p_diff = p - pp
    if p_diff > 0:
        if verbose: print(f':: p_diff > 0, adding {p_diff} new unique nodes to H')
        HC = np.where((Ho | H_fs | Ho_fs) == False)[0]
        idxs = np.random.choice(HC, size=p_diff, replace=False)
        H[idxs] = True
        pp = pp + p_diff
        assert (H & H_fs).sum() == 0
        assert k == pp + (k-p)
    
    k_diff = k - (H | H_fs).sum()
    if k_diff > 0:
        if verbose: print(f':: k_diff > 0, adding {k_diff} nodes from Ho to H')
        Hu = set(np.where(H | H_fs)[0])
        HC = set(np.where(Ho)[0]) - Hu
        if len(HC) < k_diff:
            HC |= (set(np.where(Ho_fs)[0]) - Hu)
            if len(HC) < k_diff:
                HC = np.where((Ho | Ho_fs | H | H_fs) == False)[0]
        idxs = np.random.choice(list(HC), size=k_diff, replace=False)
        assert len(set(idxs) & set(np.where(H | H_fs)[0])) == 0
        H[idxs] = True
    
    xis = list(set(np.where(Ho | Ho_fs)[0]) - set(np.where(H | H_fs)[0]))
    xjs = list(set(np.where(H | H_fs)[0]) - set(np.where(Ho | Ho_fs)[0]))
    assert (H | H_fs).sum() == k, f'combined size is {(H | H_fs).sum()} vs {k}'
    assert (H & H_fs).sum() == 0
    assert len(xis) == len(xjs), 'Numbers of removed nodes ({}) and ' \
                                 'added ({}) nodes do not match.' \
                                 .format(len(xis),len(xjs))
    return H, H_fs, xis, xjs

@njit
def shake(A: np.array, Ho: np.array, k: int, p: int, alpha: np.array, beta: np.array,
          p_w: np.array=None, use_pref_attachment=False):
    """
    Implements the perturbation routine for the VNS (variable neighborhood search)
    H by randomly drawing p node ids from the H without replacement and replacing them 
    with p randomly drawn node ids from the complement of H.
    
    Parameter use_pref_attachment controls for the weighting of the random distribution. 
    If True allows for preferential weighting based on vector of probabilities. Defaults 
    to False which equals uniform distribution over all nodes in the network.
    """
    n = A.shape[0]
    HC = np.where(~Ho)[0]
    if use_pref_attachment and p_w is not None:
        p_w_norm = p_w[HC] / p_w[HC].sum()
    else:
        p_w_norm = None
    xis = choice(np.where(Ho)[0], size=p)
    xjs = choice(HC, size=p, p=p_w_norm)
    H = Ho.copy()
    H[xis] = False
    H[xjs] = True
    alpha_p = alpha.copy() 
    beta_p = beta.copy()
    for i in range(p):
        xi = xis[i]
        xj = xjs[i]
        alpha_p, beta_p = __update_degree_vecs(A, alpha_p, beta_p, 
                                               xi, xj, inplace=True)
    H_w = alpha_p[H].sum() / 2
    return H, H_w, alpha_p, beta_p

def shake_old(A: np.array, Ho: np.array, k: int, p: int, alpha: np.array, beta: np.array,
          p_w: np.array=None, use_pref_attachment=False, verbose=False):
    """Implements the perturbation routine for the VNS (variable neighborhood search)
    H by randomly drawing p node ids from the H without replacement and replacing them 
    with p randomly drawn node ids from the complement of H.
    
    Parameter use_pref_attachment controls for the weighting of the random distribution. 
    If True allows for preferential weighting based on vector of probabilities. Defaults 
    to False which equals uniform distribution over all nodes in the network.
    """
    n = A.shape[0]
    HC = np.where(~Ho)[0]
    xis = np.random.choice(np.where(Ho)[0], size=p, replace=False)
    xjs = np.random.choice(HC, size=p, replace=False, p=p_w[HC] / p_w[HC].sum() 
                           if use_pref_attachment else None)
    H = Ho.copy()
    H[xis] = False
    H[xjs] = True
    alpha_p = alpha.copy() 
    beta_p = beta.copy()
    for i in range(p):
        xi = xis[i]
        xj = xjs[i]
        alpha_p, beta_p = __update_degree_vecs(A, alpha_p, beta_p, 
                                               xi, xj, inplace=True)
    H_w = alpha_p[H].sum() / 2
    if verbose: 
        print(f':: Updated {p} nodes {xis} with {xjs}.')
        blocksum = A[H,:][:,H].sum() / 2
        print(f':: Alpha based H_w: {H_w:.4f}, {blocksum:.4f}')
        assert (blocksum - H_w) < 1e-2 
    return H, H_w, alpha_p, beta_p

def shake_fs(A: np.array, Ho: np.array, Ho_fs: np.array, fss: list, n_ss: dict, k: int, 
             p: int, p_max, alpha: np.array, beta: np.array, verbose=False):
    """Implements the perturbation routine for the VNS when forced selection is used.
    
        Method has two different behaviors that are dependent on the p value and H_fs size:
        1. when p is smaller than the smallest available H_fs, only nodes in H are swapped.
        2. when p is at least the size of the smallest available H_{fs} configuration, 
           H_fs configuration and p - |H_fs| nodes from H are swapped.
    """
    assert k > Ho_fs.sum(), 'size of H_fs is larger than k, try increasing k'
    n = A.shape[0]
    c_min = min(n_ss.keys())
    c_max = max(n_ss.keys())
    
    if p < c_min and p > Ho.sum():
        p = np.random.randint(c_min, np.min([c_max, p_max]))
    if p >= c_min:
        repeat_count = 0
        while True: 
            H, H_fs, xis, xjs = __replace_in_hhfs(Ho, Ho_fs, fss, n_ss, k, n, p, verbose)
            if (H_fs & Ho_fs).sum() != Ho_fs.sum():
                break
            repeat_count+=1
            assert repeat_count < 5, 'Limited number of choices in the fixed set ' \
                                     ' configurations, try generating more.'
    else:
        H, H_fs, xis, xjs = __replace_in_h(Ho, Ho_fs, p)
    
    alpha_p = alpha.copy()
    beta_p = beta.copy()
    for i in range(len(xis)):
        xi = xis[i]
        xj = xjs[i]
        alpha_p, beta_p = __update_degree_vecs(A, alpha_p, beta_p, 
                                               xi, xj, inplace=True)
    Hfs_len = (H | H_fs).sum()
    assert Hfs_len == k, f'H combined size {Hfs_len} does not match with k={k}'
    assert (H & H_fs).sum() == 0, 'There is overlap in the selection\n * : ' \
                                          '{}'.format(set(H_fs) & set(H))
    H_w = alpha_p[H | H_fs].sum() / 2
    return H, H_w, H_fs, alpha_p, beta_p
