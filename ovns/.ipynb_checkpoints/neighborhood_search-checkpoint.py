from time import process_time
from datetime import timedelta as td
import numpy as np
from numba import *
from ovns.initialize import *
from ovns.utils import __update_degree_vecs

@njit
def ls_one_n_beam(Uo, Uo_w, A, A_beam, alpha, beta, tol=0.0, 
                  find_maxima=False, one_in_k=False, verbose=False):
    """Computes local search in the 1-neighborhood of the Ho set such that 
    node is selected using beam criterion; the space of neighbors with heaviest 
    links in the 1-neighborhood of the current H nodes is searched first. First 
    improvement of the objective function is returned.
    """
    k = Uo.sum()
    n = A_beam.shape[1]
    
    Up_w = Uo_w
    f_prime = 0.0
    xip = xjp = -1
    
    u_idxs = np.where(Uo)[0]
    if not one_in_k:
        replace_ids = u_idxs.copy()
    if not find_maxima:
        np.random.shuffle(u_idxs)
        
    L = 0
    
    stop = False
    for i in range(k):
        if stop: break
        v = u_idxs[i]
        for j in range(n):
            if stop: break
            xj = A_beam[v,j]
            if xj != -1 and not Uo[xj]:
                if one_in_k:
                    replace_ids = np.random.choice(u_idxs, 1)
                for xi in replace_ids:
                    L += 1
                    delta_f = alpha[xj] - alpha[xi] - A[xi,xj]
                    if delta_f > f_prime:
                        Up_w = Uo_w + delta_f
                        f_prime = delta_f
                        xip = xi
                        xjp = xj
                        if verbose:
                            print(':: Improvement found: +', (delta_f))
                            print(':: Objective function value: ', Up_w,', iters: ', L)
                        if not find_maxima:
                            stop = True
                            break
    if Up_w == Uo_w:
        if verbose: print(':: No improvement found during local search.')
        return Uo, Uo_w, alpha, beta
    
    assert xip >= 0 and xjp >= 0
    alpha_p, beta_p = __update_degree_vecs(A, alpha, beta, xip, xjp)
    Up = Uo.copy()
    Up[xjp] = True
    Up[xip] = False
    return Up, Up_w, alpha_p, beta_p

@njit
def ls_one_n_beam_fs(Uo, Uo_fs, Uo_w, A, A_beam, alpha, beta, tol=0.0, 
                           find_maxima=False, one_in_k=False, verbose=False):
    """Computes local search in the 1-neighborhood of the Ho set such that 
    node is selected using beam criterion; the space of neighbors with heaviest 
    links in the 1-neighborhood of the current H nodes is searched first. First 
    improvement of the objective function is returned.
    """
    k1 = Uo.sum()
    k2 = Uo_fs.sum()
    n = A_beam.shape[1]
    
    # Keep track of best improvement
    Up_w = Uo_w
    f_prime = 0.0
    xip = xjp = -1
    
    u_idxs = np.where(Uo)[0]
    u_idxs_fs = np.where(Uo_fs)[0]
    
    if not one_in_k:
        replace_ids = u_idxs.copy()
    if not find_maxima:
        np.random.shuffle(u_idxs)
        
    L = 0
    
    stop = False
    for i in range(k1+k2):
        if stop: break
        v = u_idxs[i] if i < k1 else u_idxs_fs[i-k1] 
        for j in range(n):
            if stop: break
            xj = A_beam[v,j]
            if xj != -1 and not Uo[xj] and not Uo_fs[xj]:
                if one_in_k:
                    replace_ids = np.random.choice(u_idxs, 1)
                for xi in replace_ids:
                    L += 1
                    delta_f = alpha[xj] - alpha[xi] - A[xi,xj]
                    if delta_f > f_prime:
                        Up_w = Uo_w + delta_f
                        f_prime = delta_f
                        xip = xi
                        xjp = xj
                        if verbose:
                            print(':: Improvement found: +', (delta_f))
                            print(':: Objective function value: ', Up_w,', iters: ', L)
                        if not find_maxima:
                            stop = True
                            break
    if Up_w == Uo_w:
        if verbose: print(':: No improvement found during local search.')
        return Uo, Uo_w, alpha, beta
    
    assert xip >= 0 and xjp >= 0
    alpha_p, beta_p = __update_degree_vecs(A, alpha, beta, xip, xjp)
    Up = Uo.copy()
    Up[xjp] = True
    Up[xip] = False
    return Up, Up_w, alpha_p, beta_p
