from time import process_time
from datetime import timedelta as td
import numpy as np
from numba import *
from kmmi.utils.utils import sub_sum
from ovns.initialize import *
from ovns.neighborhood_change import *
from ovns.neighborhood_search import *
from ovns.utils import __create_bvns_array

def BVNS(k: int, A: np.array, k_lims: tuple, k_step: int=1, timetol: int=300, 
        ls_tol: float=0.0, ls_mode='first', seed: int=None, max_iter: int=1000000, 
        max_iter_upd: int=100000, init_solution=None, n_init_population=1000, verbose=False): 
    """Variable neighborhood search heuristic based on (Brimberg et al. 2009) for the heaviest 
    k-subgraph problem. 'First improvement' local search mode is used by default.
    """
    n = A.shape[0]
    assert ls_mode in ['first','best'], "Invalid local search mode; choose either 'first' or 'best'" 
    assert k != n, 'Input k value equals n, solution contains all nodes in the network'
    assert k < n, 'Input k value is greater than n; select k such that k <= n'
    assert np.abs(A[0,:].sum() - A[:,0].sum()) < 1e-5, 'Directed networks are not supported'
    assert k_lims[1] <= n-k, 'Invalid upper limit of the k step size'
    
    t0 = process_time()
    Ap = __create_bvns_array(A)
    find_maxima = ls_mode == 'best'
    if k_lims is None:
        k_lims = (1, np.min([k, n-k]))
    
    m = (A > 1e-5).sum()
    print(f':: Running BVNS ({n:,} nodes, {m:,} edges, {m / (n*(n-1)):.6f} density)')
    Ho, Ho_w, ao, bo, p_w = initialize_solution(
        A, Ap, k, init_solution, find_maxima=find_maxima, 
        n_init_population=n_init_population, verbose=verbose
    )
    hss = [(Ho)]
    run_trace = [(Ho_w, 0)]
    
    best_time = delta_t0 = process_time() - t0
    print(':: Initialization and first local search completed.')
    print(':: LS took {}, value: {}\n'.format(str(td(seconds=delta_t0)), Ho_w))
    
    i = i0 = 0
    stop = False
    while not stop:
        k_cur = k_lims[0]
        while k_cur <= k_lims[1] and not stop:
            
            # 1. Perturbate
            if k_cur > 0:
                H, H_w, ap, bp = shake(A, Ho, k, k_cur, ao, bo, None, False)
                assert k - (H & Ho).sum() == k_cur, f'k={k}, |H & Ho|={(H & Ho).sum()}, p={k_cur}'
                if verbose: print(':: Perturbation @ depth ', k_cur)
            else:
                H, H_w = Ho.copy(), Ho_w
                ap, bp = ao.copy(), bo.copy()
                
            # 2. Find local improvement
            H, H_w, ap, bp = ls_one_n_beam(H, H_w, A, Ap, alpha=ap, beta=bp, 
                                           tol=ls_tol, find_maxima=find_maxima, 
                                           verbose=verbose)
            if verbose and find_maxima:
                if H_w != Ho_w: 
                    print(':: Local maxima:', H_w, '\n')
            i += 1
            i0 += 1
            if H_w > Ho_w:
                delta_w = (H_w-Ho_w) / Ho_w * 100
                print(':: Found new maxima: {:.6f}, change: +{:.2f}%'.format(H_w, delta_w))
                print(':: iteration: {}, distance in iterations to earlier update: {}'
                      .format(i, i0))
                print(50*'--')
                i0 = 0
                Ho_w = H_w
                Ho = H
                ao = ap.copy()
                bo = bp.copy()
                hss.append(Ho)
                best_time = process_time() - t0
                run_trace.append((H_w, i))
                k_cur = k_lims[0]
            else:
                k_cur += k_step
            stop = (i >= max_iter or i0 >= max_iter_upd or process_time() - t0 >= timetol)
                
    delta_t = process_time()-t0
    print(':: Run completed @ {} ({} iterations), final f value: {:.6f} ({:.6f} per node)'
          .format(str(td(seconds=delta_t)), i, Ho_w, Ho_w / k))
    
    local_maximas_h = [np.where(h)[0] for h in hss]
    
    params = {'k':k, 'k_lims':k_lims,'k_step':k_step,'timetol':timetol,'ls_tol':ls_tol,
              'ls_mode':ls_mode,'seed':seed,'max_iter':max_iter,'max_iter_upd':max_iter_upd,
              'init_solution':init_solution}
    
    run_vars = {'H':np.where(Ho)[0], 'obj_score':Ho_w, 'local_maximas_h':local_maximas_h, 
                'run_trace':run_trace, 'running_time':delta_t, 'iterations':i, 'params':params,
                'init_time':delta_t0, 'best_time':best_time}
    return run_vars
