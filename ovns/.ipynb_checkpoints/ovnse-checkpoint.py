from time import process_time
from datetime import timedelta as td
import numpy as np
import timeit
from numba import *
from ovns.ovns import *
from ovns.utils import sub_sum

def OVNSE(k: int, A: np.array, k_lims: tuple, k_step: int=1, timetol: int=300, ls_tol: float=0.0, 
          ls_mode='first', use_pref_attachment=True, init_mode='drop-initial', beta_ratio: float=0.75,
          finetuning_ratio: float=0.2, seed: int=None, max_iter: int=2000000, max_iter_upd: int=1000000, 
          w_quantile=1.0, init_solution: np.array=None, svns=False, theta: float=0.06, one_in_k=False, 
          auto_parametrize=True, verbose=False, k_step_selection='deterministic'): 
    """Opportunistic Variable Neighborhood Search Extended heuristic for the HkSP. This version of the
    algorithm splits the run into two separate phases, and their ratio is determined by 'finetuning ratio'
    parameter.
    
    Parameters
    ----------
    k : int
        Value of k in HkS (number of treatment groups).
    A : numpy.array (symmetric)
        Weighted adjacency matrix of the input network.
    k_lims : tuple of int
        Range for search depth, in general bounds are (0,k) or (0,n-k) if k > 0.5*dim(A).
        See also 'auto_parametrize'.
    k_step : int 
        Defines how many search depth steps to increment at once after unsuccessful update
        (default 1). See also 'auto_parametrize'.
    timetol : int
        Set the time upper bound (in seconds) for single run, (default 300)
    ls_tol : float, optional
        Set a tolerance for update (update is approved if improvement over current best 
        solution is at least ls_tol).
    ls_mode : string, optional
        Set local search mode. (default 'first')
        
        - 'best' : best improvement mode does exhaustive search over the 
                   space of all N1 neighbors and returns the best improvement.
        - 'first' : first improvement begins like 'best' but returns on first 
                    found improvement which allows faster exploration. Useful 
                    for large networks and high k values.
    one_in_k : bool, optional
         If True, selects one of the k nodes in the existing solution uniformly in random
         for replacement. Enabling allows more rapid exploration. See also 'auto_parametrize',
         (default False).
    use_pref_attachment : bool, optional
         When True perturbation will be biased towards high degree nodes similarly 
         as in preferential attachment in BA random network model (default True).
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
    finetuning_ratio : float, optional
         Sets the ratio of computing resources allocated for finetuning (enhanced exploitation) 
         phase at the end of the run between [0,1.0], at 0 finetuning is disabled and behavior is 
         equal to regular OVNS, (default 0.2).
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
    auto_parametrize : bool, optional
         If True, following parameters will be adjusted using simple heuristics to achieve minimum 
         speed of 1e4 optimization cycles per minute: (ls_mode, one_in_k, w_quantile, k_step), 
         (default True).
         
         Note! the auto_parametrize selection heuristic is based on early hyperparameter 
         optimization runs. However, they might not generalise to your use case. For best 
         results, it is advisable to run proper hyperparamater optimization for your 
         particular data set.
    verbose : bool
         Enables verbose mode (default False).
    k_step_selection : str, optional 
         Select one from deterministic, uniform, binomial, scaled+uniform, scaled+binomial
         Control how the k_step is applied in neighborhood change / shake operation. If scaled+uniform
         is selected, the increase will be a multiplicative function of k_step size and number of 
         unsuccessful updates. 
    
    Returns
    -------
    run_vars : dict 
         Dict with following keys:
         
         - H : numpy.array
             Best found approximate solution for the HkSP as a as k length sequence of node ids 
             (int) corresponding to indeces in the A matrix.
         - obj_score : float 
             Objective function value for best solution, f(H).
         - local_maximas_h : numpy.array with shape (x,n)
             History of all x solution updates for the executed run as boolean vectors of length n.
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
        
    """
    assert 0.0 <= finetuning_ratio <= 1.0, 'Invalid value, select value from range [0.0,1.0]'
    
    t0 = process_time()
    
    # Split time based on finetuning ratio
    finetuning_timetol = timetol*(1.0-finetuning_ratio)
    finetuning_max_iter = max_iter*(1.0-finetuning_ratio)
    
    print(f':: Initializing phase 1/2, exploration run with {100*(1-finetuning_ratio)}%' \
          ' resources allocated')
    ovns_p1 = OVNS(
        k=k, A=A, k_lims=k_lims, k_step=k_step, timetol=finetuning_timetol, 
        ls_tol=ls_tol, ls_mode=ls_mode, use_pref_attachment=use_pref_attachment, 
        init_mode=init_mode, beta_ratio=beta_ratio, seed=seed, max_iter=finetuning_max_iter, 
        max_iter_upd=max_iter_upd, w_quantile=w_quantile, init_solution=init_solution,
        svns=svns, theta=theta, one_in_k=one_in_k, auto_parametrize=auto_parametrize, 
        verbose=verbose, k_step_selection=k_step_selection
    )

    print(f':: Initializing phase 2/2, finetuning run with {100*finetuning_ratio}%' \
          ' resources allocated')
    
    # Enable finetuning
    auto_parametrize = False
    w_quantile = 1.0
    k_step = 1
    one_in_k = False
    ls_mode = 'best'
    
    init_solution = ovns_p1['H']
    print('obj_score:', ovns_p1['obj_score'])
    print('Alpha vector summed:', ovns_p1['alpha'][init_solution].sum() / 2)
    print('A row sums:', A[init_solution,:][:,init_solution].sum() / 2)
    print('A sub sums (numba):', sub_sum(A, init_solution))
    
    finetuning_timetol = timetol*finetuning_ratio
    finetuning_max_iter = max_iter*finetuning_ratio
    
    ovns_p2 = OVNS(
        k=k, A=A, k_lims=k_lims, k_step=k_step, timetol=finetuning_timetol, 
        ls_tol=ls_tol, ls_mode=ls_mode, use_pref_attachment=use_pref_attachment, 
        init_mode=init_mode, beta_ratio=beta_ratio, seed=seed, max_iter=finetuning_max_iter, 
        max_iter_upd=max_iter_upd, w_quantile=w_quantile, init_solution=init_solution,
        svns=svns, theta=theta, one_in_k=one_in_k, auto_parametrize=auto_parametrize, 
        verbose=verbose, k_step_selection=k_step_selection
    )
    
    ovns_p1['params']['finetuning_ratio'] = ovns_p2['params']['finetuning_ratio'] = finetuning_ratio
    return ovns_p1, ovns_p2
