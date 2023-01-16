# -*- coding: utf-8 -*-
from time import process_time
from datetime import timedelta as td
import numpy as np
from numba import *
from ovns.initialize import *
from ovns.neighborhood_change import *
from ovns.neighborhood_search import *
from ovns.utils import __create_beam_array, __svns_score, __to_len_classes, sub_sum
from collections import Counter

def OVNS_fs(k: int, A: np.array, fss: list, k_lims: tuple, k_step: int=1, timetol: int=300,
           ls_tol: float=0.0, ls_mode: str='best', beta_ratio: float=0.5, seed: int=None, 
           w_quantile: float=0.75, init_mode='drop-initial', svns_offset=1e5, theta: float=1e-4, 
           one_in_k=False, max_iter: int=100000, max_iter_upd: int=100000, 
           init_solution: tuple=None, verbose=False, ss = None):
    """Opportunistic Variable Neighborhood Search heuristic for the set constrained 
    variant of HkSP. 
    
    Parameters
    ----------
    k : int
        Value of k in HkS (number of treatment groups).
    A : numpy.array (symmetric)
        Weighted adjacency matrix of the input network.
    fss : list containing one dimensional sequences of type numpy.array
        Set constrained graphlet sequences
    k_lims : tuple of int
        Range for search depth, in general bounds are (0,k) or (0,n-k) if k > 0.5*dim(A).
    k_step : int 
        Defines how many search depth steps to increment at once after unsuccessful update
        (default 1).
    timetol : int
        Set the time upper bound (in seconds) for single run, (default 300)
    ls_tol : float, optional
        Set a tolerance for update (update is approved if improvement over current best 
        solution is at least ls_tol).
    ls_mode : string, optional
        Set local search mode. (default 'best')
        
        - 'best' : best improvement mode does exhaustive search over the 
                   space of all N1 neighbors and returns the best improvement.
        - 'first' : first improvement begins like 'best' but returns on first 
                    found improvement which allows faster exploration. Useful 
                    for large networks and high k values.
    one_in_k : bool, optional
         If True, selects one of the k nodes in the existing solution uniformly in random
         for replacement. Enabling allows more rapid exploration (default False).
    init_mode : string, optional
         Select initialization strategy ('drop-initial','heaviest-edge','random','weighted-deg')
         (default 'drop-initial')
         
         - 'drop-initial' : start with all n nodes in the solution and iteratively
                            remove the node that contributes least to the solution.  
         - 'heaviest-edge' : selects k-x heaviest edges such that number of 
                             nodes in the solution amount to k.
         - 'random' : fully random initialization, selects k nodes uniformly in random.
         - 'weighted-deg' : selects k nodes based on the linear combination 
                            of their degree and weighted degree ranking. 
                            'beta_ratio' adjusts the weighting.
        See also 'init_solution'.
    beta_ratio : float, optional
         Sets the weights for the `weighted-deg` initialization strategy in range
         [0,1.0] where 0 gives all weight to degree ranking and 1 to weighted 
         degree ranking, (default 0.75).
    init_solution : numpy.array or list, optional
         Initial solution as a k length sequence of node ids (int) corresponding to A indeces 
         (default None).
    seed : int, optional
         Seed for the random number generator (note: due to lack of numba support for numpy, 
         some numpy.random library calls don't allow seed input currently). (default None)
    w_quantile : float, optional
         Define the quantile of heaviest edge weights that will be explored during each 
         neighorhood search (also "beam width"), lower values increase exploration speed.
         (default 1.0). See also 'auto_parametrize'.
    svns : bool, optional
         Enable svns mode. See also 'theta' for adjusting the search sensitivity.
    theta : float, optional
         Search sensitivity for svns mode. Conditional on 'svns' parameter being True.
    max_iter : int, optional
         Set the stopping criteria as number of max iterations (also optimization cycles)
         (default 2e6).
    max_iter_upd : int, optional
         Set the convergence criteria as maximum number of iterations (also, optimization 
         cycles) after last successful update (default 1e6).
    verbose : bool
         Enables verbose mode (default False).
    
    Returns
    -------
    run_vars : dict 
         Dict with following keys:
         
         - H : numpy.array, shape is (k-len(H_fs))
             The non-set-constrained part of the HkSP solution as a k length sequence of node ids 
             (int) corresponding to indeces in the A matrix.
         - H_fs : numpy.array, shape is (k-len(H))
             Part of the best found solution that includes only the set constrained nodes, format 
             similar to H. 
         - obj_score : float 
             Objective function value for best solution, f(H).
         - local_maximas_h : numpy.array with shape (x,n)
             History of all x solution updates for the non-set-constrained part of the solution 
            as boolean vectors of length n.
         - local_maximas_h_fs : 
             History of all x solution updates for the set constrained part as boolean vectors 
             of length n.
         - run_trace : list of tuples with length x
             List with objective score function values (tuple index 0) and iteration index 
             (tuple index 1) for the executed run.
         - running_time : int
             Running time in seconds, measured as process time (not wall time).
         - iterations : int
             Number of iterations (also optimization cycles).
         - params : dict
             Dictionary that includes all run parameters.
         - converged : bool
             True if run satisfied the convergence criteria set by 'max_uter_upd'.
         
    Examples
    --------
    Simple toy example with a random 1000x1000 adjacency matrix and randomly drawn
    constrained set L.
    
    >>> n = 1000
    >>> A = np.random.random((n,n))
    >>> # Remove self-edges
    >>> A[np.diag_indices_from(A)] = 0.0

    >>> # Set OVNS params
    >>> k = 16
    >>> k_lims = (1,k)
    >>> timetol = 10

    >>> # Draw constrained set L (subset of V)
    >>> n_L = 100
    >>> L = np.random.choice(n, size=n_L, replace=False)
    
    >>> # Then draw random sequences (for serious use case you'd want 
    >>> # to compute these using the 'force_selection' module)
    >>> n_fs = 8 # Size of the set constrained part of the solution
    >>> n_sample = 100000
    >>> fss = [np.random.choice(L, size=n_fs, replace=False) for _ in range(n_sample)]
    
    >>> # Run
    >>> run = OVNS_fs(k, A, fss, k_lims, timetol=timetol)
    :: Initial local search completed.
    :: LS took 0:00:06.218111, value: 69.22395116412441 
    
    :: Found new maxima: 70.859469, change: +2.36%
    :: iteration: 1, distance in iterations to earlier update: 1
    ------------------------------------------------------------------
    :: Found new maxima ... ... ... 
    ------------------------------------------------------------------
    :: Found new maxima: 71.821457, change: +1.36%
    :: iteration: 2, distance in iterations to earlier update: 1
    ------------------------------------------------------------------
    :: Found new maxima: 106.826962, change: +0.29%
    :: iteration: 4641, distance in iterations to earlier update: 65
    ------------------------------------------------------------------
    :: Run completed @ 0:00:10.001934 (4765 iterations)
    :: final f value: 106.826962 (6.676685 per node)
    
    >>> print('Selected params: ', run['params'])
    Selected params:  {'k': 16, 'k_lims': (1, 16), 'k_step': 1, 'timetol': 10, 
                       'ls_tol': 0.0, 'ls_mode': 'best', 'init_mode': 'drop-initial', 
                       'beta_ratio': 0.5, 'w_quantile': 0.75, 'seed': None, 
                       'max_iter': 100000, 'max_iter_upd': 100000, 'init_solution': None, 
                       'theta': 0.06, 'svns': False, 'one_in_k': False}
                       
    >>> print('Best, non-constrained set:', run['H'])
    Best, non-constrained set: [ 18  99 216 227 515 551 914 943]
    
    >>> print('Best, constrained set :', run['H_fs'])
    Best, constrained set : [ 49  56 200 480 708 758 986 987]

    >>> print('Best f value: ', run['obj_score'])
    Best f value:  106.8269618027097
    """
    n = A.shape[0]
    CONTENT_END_CHAR = -1
    assert ls_mode in ['first','best'], "Invalid local search mode; choose either " \
                                        " 'first' or 'best'" 
    assert k != n, 'Input k value equals n, solution contains all nodes in the network'
    assert k < n, 'Input k value is greater than n; select k such that k < n'
    assert 0.0 < w_quantile <= 1.0, 'Improper value, select value from range (0.0,1.0]'
    if k_lims[1] > n-k:
        print(':: WARNING: upper limit {} of the k_lims is above the available pool ' \
              'of values.'.format(k_lims[1]))
        k_lims = (k_lims[0], np.min([k, n-k]))
        print('::      --> readjusted to {}.'.format(k_lims))
    
    n_ss = __to_len_classes(fss)
    assert k >= min(n_ss.keys()), 'Input k value is below shortest available ' \
                                  'seed node configuration length'
    
    find_maxima = ls_mode == 'best'
    w_thres = np.quantile(A[A > 0.0], 1-w_quantile) if w_quantile != 1.00 else 1e-6
    A_as = np.argsort(A)[:,::-1]
    A_beam, mu_beam_width = __create_beam_array(A, A_as, w_thres)
    assert A_beam.shape[1] > 0, 'Set w_quantile is too small, A_beam has no elements'
    if verbose and A_beam.shape[1] < 10:
        print(':: WARNING: determined beam width is narrow @ {}, optimization result ' \
              'might suffer, for better result try increasing w_quantile'
              .format(A_beam.shape[1]))
    
    hss = []
    hfs = []
    run_trace = []

    t0 = process_time()
    rsums = np.sum(A, axis=1)
    p_w = rsums / rsums.sum()
    if init_solution is None:
        if init_mode == 'drop-initial':
            H, Ho_fs, _, _ = init_solution_drop_initial_fs(A, k, fss)
        elif init_mode == 'weighted-deg':
            H, Ho_fs, _, = init_solution_weighted_degree_ranking_fs(A, k, fss, beta_ratio)
        else:
            H, Ho_fs = init_random_fs(A, k, fss)
        _, ao, bo = initialize_degree_vecs(A, H | Ho_fs)
        H_w = ao[H | Ho_fs].sum() / 2
        Ho, Ho_w, ao, bo = ls_one_n_beam_fs(H, Ho_fs, H_w, A, A_beam, ao, bo, 
                                            tol=ls_tol, find_maxima=find_maxima,
                                            one_in_k=one_in_k, verbose=verbose)    
    else:
        Ho = np.zeros(n, dtype=bool)
        Ho_fs = np.zeros(n, dtype=bool)
        Ho[init_solution[0]] = True
        Ho_fs[init_solution[1]] = True
        assert (Ho | Ho_fs).sum() == k
        assert (Ho & Ho_fs).sum() == 0
        _, ao, bo = initialize_degree_vecs(A, Ho | Ho_fs)
        H_w = Ho_w = ao[Ho | Ho_fs].sum() / 2
        
    hss.append(Ho)
    hfs.append(Ho_fs)
    run_trace.append((H_w, 0))   
        
    delta_t = process_time()-t0
    print(':: Initial local search completed.')
    print(':: SVNS update offset: {} iterations'.format(svns_offset))
    print(':: LS took {}, value: {}\n'.format(str(td(seconds=delta_t)), Ho_w))
    hfs_cond = lambda H_fs, Ho_fs: (H_fs & Ho_fs).sum() != Ho_fs.sum()
    
    i = i0 = ifs = 0
    stop = False
    n_overlap = 0
    ccc = Counter()
    n_overlaps = []
    while not stop:
        k_cur = k_lims[0]
        while k_cur <= k_lims[1] and not stop:
            
            # 1. Perturbate    
            if k_cur > 0:
                H, H_w, H_fs, ap, bp = shake_fs(A, Ho, Ho_fs, fss, n_ss, k, k_cur, k_lims[1], ao, bo)
                # Check if hfs has been visited
                if hfs_cond(H_fs, Ho_fs):
                    hfs_id = hash(frozenset(np.where(H_fs)[0]))
                    ccc.update([hfs_id])
                    
                if verbose: print(':: Perturbation @ depth ', k_cur)
            else:
                H, H_fs, H_w = Ho.copy(), Ho_fs.copy(), Ho_w
                ap, bp = ao.copy(), bo.copy()
                
            # 2. Find local improvement
            H, H_w, ap, bp = ls_one_n_beam_fs(H, H_fs, H_w, A, A_beam, alpha=ap, beta=bp, 
                                              tol=ls_tol, find_maxima=find_maxima,
                                              one_in_k=one_in_k, verbose=verbose)
            if verbose and find_maxima:
                if H_w != Ho_w:
                    print(':: Local maxima:', H_w, '\n')
            i += 1
            i0 += 1
            ifs += 1
            
            if ifs > svns_offset and n_overlap > 0 and len(n_overlaps) > 5:
                mean_nol = np.mean([np.abs(x - n_overlap) for x in n_overlaps[-5:]])
                if mean_nol < 2:
                    svns = True
                    n_overlaps.clear()
                    k_cur = max(n_ss.keys()) if max(n_ss.keys()) < k_lims[1] else k_lims[1]
                    Ho_w = np.max([hw for hw,i in run_trace])
                    print(f':: Due to overlap and  SVNS update condition enabled with θ: {theta:.4f}')
                else:
                    svns = False
            else:
                svns = False

            svns_cond = __svns_score(H_w, Ho_w,
                                     H | H_fs, Ho | Ho_fs,
                                     k) > 1 + theta if svns else False
            if H_w > Ho_w or svns_cond:
                delta_w = (H_w-Ho_w) / Ho_w * 100
                print(':: Found new maxima: {:.6f}, change: +{:.4f}%'.format(H_w, delta_w))
                print(':: iteration: {}, distance in iterations to ' \
                      'last update: {}, to last L update: {}'.format(i, i0, ifs))
                if hfs_cond(H_fs, Ho_fs):
                    ifs = 0
                if ss is not None:
                    H_nodes = [v for s in np.where(H)[0] for v in ss[s]]
                    Hfs_nodes = [v for s in np.where(H_fs)[0] for v in ss[s]]
                    overlap_counter = Counter(H_nodes + Hfs_nodes)
                    if CONTENT_END_CHAR in overlap_counter:
                        overlap_counter.pop(CONTENT_END_CHAR)
                    n_overlap = len([c for c in overlap_counter.values() if c > 1])
                    n_overlaps.append(n_overlap)
                    if n_overlap > 0:
                        print(f':: Warning, update with {n_overlap} overlaps accepted.')
                print(50*'--')
                i0 = 0
                Ho_w = H_w
                Ho = H.copy()
                Ho_fs = H_fs.copy()
                ao = ap.copy()
                bo = bp.copy()
                k_cur = k_lims[0]
                hss.append(Ho)
                hfs.append(Ho_fs)
                run_trace.append((H_w, i))
            else:
                k_cur += k_step
            stop = (i >= max_iter or i0 >= max_iter_upd or process_time() - t0 >= timetol)
    
    max_idx = np.argmax([hw for hw,i in run_trace])
    Ho_w = run_trace[max_idx][0]
    Ho = hss[max_idx]
    Ho_fs = hfs[max_idx]
    
    delta_t = process_time()-t0
    print(':: Run completed @ {} ({} iterations), final f value: {:.6f} ({:.6f} per node)'
          .format(str(td(seconds=delta_t)), i, Ho_w, Ho_w / k))
    
    local_maximas_h = [np.where(h)[0] for h in hss]
    local_maximas_h_fs = [np.where(h)[0] for h in hfs]
    
    params = {'k':k,'k_lims':k_lims,'k_step':k_step,'timetol':timetol,
              'ls_tol':ls_tol,'ls_mode':ls_mode,'init_mode':init_mode,'beta_ratio':beta_ratio,
              'w_quantile':w_quantile,'seed':seed,'max_iter':max_iter,'max_iter_upd':max_iter_upd,
              'init_solution':init_solution,'theta':theta,'svns':svns,'one_in_k':one_in_k}
    
    converged = (i0 >= max_iter_upd)
    run_vars = {'H':np.where(Ho)[0], 'H_fs':np.where(Ho_fs)[0], 'obj_score':Ho_w,
                'local_maximas_h':local_maximas_h, 'local_maximas_h_fs':local_maximas_h_fs, 
                'run_trace':run_trace, 'running_time':delta_t, 'iterations':i, 'params':params,
                'converged':converged, 'fs_counter':ccc}
    
    return run_vars
