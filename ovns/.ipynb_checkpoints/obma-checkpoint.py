import sys
import time
from time import process_time
import numpy as np
from numba import *
from datetime import timedelta as td
from ovns.utils import sub_sum, choice

@njit
def calculate_rank(ascending: bool, n: int, a):
    c = a.copy()
    b = np.arange(len(a)) + 1
    flag = np.zeros(n, dtype=np.int64)
    
    for i in range(n):
        for jc in range(n-i):
            j = (n-1)-jc
            if ascending:
                # lower score, lower rank
                if(c[j-1] > c[j]):
                    temp = c[j-1]
                    c[j-1] = c[j]
                    c[j] = temp

            else:
                # higher score, lower rank
                if(c[j-1] < c[j]):
                    temp = c[j-1]
                    c[j-1] = c[j]
                    c[j] = temp

    for i in range(n):
        for j in range(n):
            if flag[j] == 0 and a[i] == c[j]:
                b[i] = j + 1
                flag[j] = 1
                break
    return b

@njit
def find_optimal_moves(A, xis, xjs, gain, tenure, i_iter, 
                       best_swap_X, best_swap_Y, 
                       best_tabu_swap_X, best_tabu_swap_Y,
                       n_best, n_tabu_best, max_n_moves):

    best_delta = best_tabu_delta = -np.inf
    n_best = n_tabu_best = 0

    for i in range(len(xis)):
        to_remove = xis[i]
        for j in range(len(xjs)):
            to_add = xjs[j]

            delta = gain[to_add] - gain[to_remove] - A[to_remove, to_add]
            if tenure[to_remove] <= i_iter and tenure[to_add] <= i_iter:
                if delta > best_delta:
                    best_delta = delta
                    best_swap_X[0] = to_remove
                    best_swap_Y[0] = to_add
                    n_best = 1
                elif (delta == best_delta) and (n_best < max_n_moves):
                    best_swap_X[n_best] = to_remove
                    best_swap_Y[n_best] = to_add
                    n_best+=1
            else:
                if delta > best_tabu_delta:
                    best_tabu_delta = delta
                    best_tabu_swap_X[0] = to_remove
                    best_tabu_swap_Y[0] = to_add
                    n_tabu_best = 1

                elif (delta == best_tabu_delta) and (n_tabu_best < max_n_moves):
                    best_tabu_swap_X[n_tabu_best] = to_remove
                    best_tabu_swap_Y[n_tabu_best] = to_add
                    n_tabu_best+=1
    return best_swap_X, best_swap_Y, best_tabu_swap_X, best_tabu_swap_Y, \
            best_delta, best_tabu_delta, n_best, n_tabu_best

@njit
def tabu_search(h_prime, A, n, t, t0, max_n_moves, max_iter, 
                alpha, scale_factor, timetol, verbose=True):
    
    # Adapted based on Zhou et al 2017 c++ implementation
    k = np.sum(h_prime)
    used_time = min_gain_in_S = max_gain_out_S = 0.0
    i_iter = no_improve_iter = size_X = size_Y = to_remove = 0

    max_A = np.max(A)
    max_integer = np.inf
    delta = 0.0            # gain for move
    best_delta = 0.0       # best move gain for nontabu move
    best_tabu_delta = 0.0  # best move gain for tabu move
    n_best = 0             # n of best non-tabu moves
    n_tabu_best = 0        # n of best tabu moves
    
    xis = np.zeros(k, dtype=np.int64)
    xjs = np.zeros(n-k, dtype=np.int64)
    best_swap_X = np.zeros(max_n_moves, dtype=np.int64)
    best_swap_Y = np.zeros(max_n_moves, dtype=np.int64)
    best_tabu_swap_X = np.zeros(max_n_moves, dtype=np.int64)
    best_tabu_swap_Y = np.zeros(max_n_moves, dtype=np.int64)

    gain = np.zeros(n, dtype=np.float64)
    tenure = np.zeros(n, dtype=np.int64)
    x1 = x2 = 0

    # Calc potential contribution of each item to the objective function
    idxs = np.where(h_prime)[0]
    for i in range(n):
        nsum = A[i][idxs]
        gain[i] = np.sum(nsum)

    # Calc f value of incumbent solution
    h_w_prime = gain[h_prime].sum() / 2
    
    # Update improved solution
    h_w_improved = h_w_prime
    h_improved = h_prime.copy()
    improved_time = 0 # Cannot call process time inside numba, keep static

    while i_iter < max_iter:
        output = "TS: iteration", i_iter, ", best cost =", h_w_improved, ", incumbent cost=", h_w_prime
        vprint(output, verbose)

        # X: the set of items with small gain (i.e., no more than dMinInS + dmax) in S
        min_gain_in_S = np.min(gain[h_prime])
        min_gain_in_S += scale_factor * max_A
        xis = np.where((gain <= min_gain_in_S) & h_prime)[0]

        # Y: the set of items with large gain (i.e., no less than dMaxOutS - dmax) out of S
        max_gain_out_S = -1 * max_integer
        max_gain_out_S = np.max(gain[~h_prime])
        max_gain_out_S -= scale_factor * max_A
        xjs = np.where((gain >= max_gain_out_S) & ~h_prime)[0]

        # Find out all best tabu or non-tabu moves
        best_swap_X, best_swap_Y, best_tabu_swap_X, best_tabu_swap_Y, best_delta,\
        best_tabu_delta, n_best, n_tabu_best = find_optimal_moves(A, xis, xjs, gain, \
                                         tenure, i_iter, best_swap_X, best_swap_Y, 
                                         best_tabu_swap_X, best_tabu_swap_Y, n_best, 
                                         n_tabu_best, max_n_moves)
        
        # Accept a best tabu or non-tabu move */
        if ((n_tabu_best > 0) and (best_tabu_delta > best_delta) and (best_tabu_delta + h_w_prime > h_w_improved)) or n_best == 0:
            index = np.random.randint(n_tabu_best)
            to_remove = best_tabu_swap_X[index]
            to_add = best_tabu_swap_Y[index]
            h_w_prime += best_tabu_delta
        else:
            index = np.random.randint(n_best)
            to_remove = best_swap_X[index]
            to_add = best_swap_Y[index]
            h_w_prime += best_delta

        # Make a move and update */
        h_prime[to_add] = True
        h_prime[to_remove] = False

        for i in range(n):
            gain[i] += A[i,to_add] - A[i,to_remove]

        tenure[to_remove] = i_iter + determine_tabu_tenure(i_iter, t, alpha)
        tenure[to_add] = i_iter + round(0.7*determine_tabu_tenure(i_iter, t, alpha))

        # Keep the best solution found so far
        if h_w_prime > h_w_improved:
            tol = np.abs(h_w_prime - sub_sum(A, np.where(h_prime)[0]))
            h_improved = h_prime.copy()
            h_w_improved = h_w_prime
            no_improve_iter = 0
        else:
            no_improve_iter += 1
        
        i_iter+=1
    return h_w_improved, h_improved, improved_time

@njit
def determine_tabu_tenure(i_iter, t, alpha):
    """Determine tabu tenure (alpha parameter)."""
    temp = i_iter % t
    if 700 < temp <= 800:
        return 8 * alpha
    elif 300 < temp <= 400 or 1100 < temp <= 1200:
        return 4 * alpha
    elif 100 < temp <= 200 or 500 < temp <= 600 or 900 < temp <= 1000 or 1300 < temp <= 1400:
        return 2 * alpha
    else:
        return alpha

@njit
def calculate_h_distance(x1, x2, k, pop):
    u = v = 0
    sharing = 0
    while u < k and v < k:
        if pop[x1,u] == pop[x2,v]:
            sharing += 1
            u += 1
            v += 1
        elif pop[x1,u] < pop[x2,v]:
            u += 1
        elif pop[x1,u] > pop[x2,v]:
            v += 1
    distance = 1 - sharing / k
    return distance

# Create an offspring and its opposition by crossover and assign the remaining items greedily
@njit
def crossover_with_greedy(A, pop, n, k, n_parents, ps, is_obl, epsilon=1e-3):
    is_choose_p = np.zeros(ps, dtype=np.int64)
    p = np.zeros(n_parents, dtype=np.int64)
    offspring = np.zeros(k, dtype=np.int64)
    index_best_v = np.zeros(n-k, dtype=np.int64)
    n_remaining_v = np.zeros(n_parents, dtype=np.int64)
    remaining_v = np.zeros((n_parents,k), dtype=np.int64)

    n_p = 0
    while n_p < n_parents:
        choose_p = np.random.randint(ps)
        if is_choose_p[choose_p] == 0:
            p[n_p] = choose_p
            n_p+=1
            is_choose_p[choose_p] = 1

    # Build a partial solution S0 by preserving the common elements
    # we take n_parents instances randomly, and preserve common elements
    common_elem = pop[p[0]]
    for i in range(1,n_parents):
        common_elem = (common_elem & pop[p[i]])
    h_prime = common_elem.copy()
    
    # S1/S0 and S2/S0
    for i in range(n_parents):
        n_remaining_v[i] = 0
        idxs = np.where(pop[p[i]])[0]
        for j in range(k):
            v = idxs[j]
            if not h_prime[v]:
                remaining_v[i][n_remaining_v[i]] = v
                n_remaining_v[i]+=1
                h_prime[v] = 0
                
    # S0
    n_added_v = 0
    for i in range(n):
        if h_prime[i]:
            offspring[n_added_v] = i
            n_added_v+=1

    # generate an offspring by completing the partial solution in a greedy way
    while n_added_v < k:
        index_p = np.random.randint(n_parents)
        max_v_profit = -np.inf
        for i in range(n_remaining_v[index_p]):
            v_profit = 0.0
            for j in range(n_added_v):
                v_profit += A[remaining_v[index_p][i], offspring[j]]

            if v_profit > max_v_profit:
                max_v_profit = v_profit
                index_best_v[0] = i
                n_best_v = 1
            elif np.abs(v_profit-max_v_profit) < epsilon:
                index_best_v[n_best_v] = i
                n_best_v+=1

        index_remaining_v = index_best_v[np.random.randint(n_best_v)]
        choose_v = remaining_v[index_p][index_remaining_v]

        offspring[n_added_v] = choose_v
        n_added_v += 1
        h_prime[choose_v] = True
        n_remaining_v[index_p] -= 1
        remaining_v[index_p][index_remaining_v] = remaining_v[index_p][n_remaining_v[index_p]]

    # Generate an opposite solution
    if is_obl:
        opposite_h = np.zeros(n, dtype=np.bool_)
        if k < n-k:
            idxs = np.where(~h_prime)[0]
            idxs_available_v = choice(idxs, size=k, replace=False)
            opposite_h[idxs_available_v] = True
            
        else:
            # Select non-overlapping part
            opposite_h[~h_prime] = True
            if k > n-k:
                # Select overlapping part
                idxs = np.where(h_prime)[0]
                idxs_available_v = choice(
                    idxs, size=k-opposite_h.sum(), replace=False
                )
                opposite_h[idxs_available_v] = True

    h = np.zeros(n, dtype=np.bool_)
    h[offspring] = True
    return h, opposite_h

@njit
def rank_based_pool_updating(h_prime, h_w_prime, alpha, pop, 
                             pop_cost, h_distance, ps, k):
    pop_distance = np.zeros(ps+1, dtype=np.float64)
    pop_score = np.zeros(ps+1, dtype=np.float64)
    
    # 1. Introduce offspring into the population
    pop_cost[ps] = h_w_prime
    pop[ps] = h_prime

    for i in range(ps):
        h_distance[i][ps] = calculate_h_distance(i,ps,k,pop)
        h_distance[ps][i] = h_distance[i][ps]
    h_distance[ps][ps] = 0.0

    # 2. For x in pop, compute mean distance wrt population
    for i in range(ps+1):
        avg_h_distance = 0.0
        for j in range(ps+1):
            if j != i:
                avg_h_distance += h_distance[i][j]
        pop_distance[i] = avg_h_distance/ps

    # 3. Compute cost and distance ranking
    cost_rank = calculate_rank(True, ps+1, pop_cost)
    distance_rank = calculate_rank(False, ps+1, pop_distance)
                      
    # 4. For x in pop, compute the combined score
    for i in range(ps+1):
        pop_score[i] = alpha*cost_rank[i] + (1.0-alpha)*distance_rank[i]
    
    min_score = np.inf
    for i in range(ps+1):
        if pop_score[i] < min_score:
            min_score = pop_score[i]
            index_worst = i

    # Insert the offspring
    is_duplicate = is_duplicate_sol(pop, h_prime, k, ps)
    if index_worst != ps and not is_duplicate:
        pop_cost[index_worst] = h_w_prime
        pop[index_worst] = h_prime

        for i in range(ps):
            h_distance[i][index_worst] = h_distance[ps][i]
            h_distance[index_worst][i] = h_distance[i][index_worst]
        h_distance[index_worst][index_worst] = 0.0

    return pop, pop_cost, h_distance

@njit
def record_best_solution(h_w_prime, h_prime, h_opt, h_w_opt, improved_time, best_time):
    """Updates the values of h_opt, and best_time if h_w_prime
    is greater than h_w_opt.
    
    This method compares the values of h_w_prime and h_w_opt. If h_w_prime is
    greater than h_w_opt, it updates the values of h_w_opt, h_opt, 
    and best_time. If h_w_prime is not greater than h_w_opt, it
    increments no_improve_gen by 1.
    
    Args:
        h_w_prime (float): The current value of h_w_prime.
        h_w_opt (float): The current value of h_w_opt.
        no_improve_gen (int): The current value of no_improve_gen.
        improved_time (float): The current value of improved_time.
    
    Returns:
        tuple: A tuple containing the updated values of h_opt, 
               and best_time if h_w_prime is greater than h_w_opt, or a tuple 
               containing the current values of h_opt and no_improve_gen if 
               h_w_prime is not greater than h_w_opt.
    """
    delta_w = round((h_w_prime-h_w_opt) / h_w_opt * 100, 2)
    h_w_opt_rounded = round(h_w_prime, 6)
    print(':: Found new maxima:', h_w_opt_rounded, '| change: +', delta_w ,'%')
    h_w_opt = h_w_prime
    h_opt = h_prime
    best_time = improved_time
    return h_opt, h_w_opt, best_time

@njit
def is_duplicate_sol(pop, h_improved, k, index):
    for i in range(index):
        if (h_improved & pop[i]).sum() == k:
            return True
    return False

@njit
def vprint(output, verbose):
    if verbose: print(':: ', output)

# Initialize the population with opposition-based learning
@njit
def opposition_based_initialization(A, k, n_parents, n, ps, t, t0, max_n_moves,
            max_iter, alpha, scale_factor, timetol, epsilon=1e-3, verbose=True):
    """
    Initialize a population of binary solutions using opposition-based learning.

    Parameters:
    - A (NumPy array): A 2D NumPy array representing the input matrix.
    - k (int): The number of ones in the solution.
    - n_parents (int): The number of parents to use in the opposition-based learning.
    - n (int): The number of bits in the solution.
    - ps (int): The size of the population.
    - t (int): The maximum number of iterations in the Tabu search.
    - h_opt (NumPy array): A 1D NumPy array representing the current optimal solution.
    - h_w_opt (float): The cost of the current optimal solution.
    - t0 (int): The initial size of the Tabu list.
    - max_n_moves (int): The maximum number of moves in the Tabu search.
    - epsilon (float, optional): The convergence threshold for the Tabu search. Default is 1e-3.
    - verbose (bool): enabling gives more information, defaults to False.

    Returns:
    - h_opt (NumPy array): A 1D NumPy array representing the optimal solution found.
    - h_w_opt (float): The cost of the optimal solution.
    - h_distance (NumPy array): A 2D NumPy array representing the distance matrix between 
      all pairs of solutions in the population.
    - pop (NumPy array): A 2D NumPy array representing the population of solutions.
    - pop_cost (NumPy array): A 1D NumPy array representing the cost of each solution in the population.
    - best_time (int): Represents the relative time it took to find the optimal solution.
    """
    h_improved = None
    h_w_improved = 0.0
    h_distance = np.zeros((ps+1, ps+1), dtype=np.float64)
    
    pop = np.zeros((ps+1, n), dtype=np.int64)
    pop_cost = np.zeros(ps+1, dtype=np.float64)
    best_time = np.inf
    h_w_opt = 0
    h_opt = np.zeros(n, dtype=np.bool_)
    
    n_h = 0
    pool = np.arange(n)
    while n_h < ps: #or process_time() - t0 > timetol:
        vprint('Iteration: ' + str(n_h), verbose)
        vprint('# 1. Generate solution candidate', verbose)
        h_prime = np.zeros(n, dtype=np.bool_)
        idxs = choice(pool, size=k, replace=False)
        h_prime[idxs] = True
    
        h_w_prime, h_prime, improved_time = tabu_search(
            h_prime, A, n, t, t0, max_n_moves, max_iter, 
            alpha, scale_factor, timetol, 
            verbose=verbose
        )

        vprint('# 2. Generate opposite solution candidate', verbose)
        opp_h = np.zeros(n, dtype=np.bool_)
        if k <= n-k:
            idxs = choice(np.where(~h_prime)[0], size=k, replace=False)
        else:
            opp_h[~h_prime] = True
            idxs = choice(np.where(h_prime)[0], size=2*k-n, replace=False)
        opp_h[idxs] = True
        
        h_opp_w_improved, h_opp_improved, improved_time = tabu_search(
            opp_h, A, n, t, t0, max_n_moves, max_iter, 
            alpha, scale_factor, timetol, verbose=verbose
        )

        vprint('# 3. Select better solution', verbose)
        if h_opp_w_improved > h_w_prime:
            h_prime = h_opp_improved.copy()
            h_w_prime = h_opp_w_improved
        
        # 4. If duplicate, alter
        while True:
            vprint(':: Found duplicate solution, modifying...', verbose)
            index = np.random.randint(k)
            swapout_v = np.where(h_prime)[0][index]
            swapout_gain = np.sum(A[h_prime, swapout_v])
            index = np.random.randint(n-k)
            swapin_v = np.where(~h_prime)[0][index]            
            swapin_gain = np.sum(A[h_prime, swapin_v])
            h_prime[swapin_v] = True
            h_prime[swapout_v] = False
            h_w_prime += (swapin_gain - swapout_gain)
            is_duplicate = is_duplicate_sol(pop, h_prime, k, n_h)
            if not is_duplicate:
                break
        pop_cost[n_h] = h_w_prime
        pop[n_h] = h_prime.copy()

        n_h += 1
        if h_w_prime > h_w_opt:
            #output = ':: Found new maxima: ', h_w_prime, ', time: ', best_time)
            #print(output, verbose)
            h_w_opt = h_w_prime
            h_opt = h_prime.copy()
            best_time = improved_time

    # Calculate the distance between any two solutions in the population
    for i in range(ps):
        for j in range(ps):
            h_distance[j][i] = h_distance[i][j] = calculate_h_distance(i, j, k, pop)
        
    return h_opt, h_w_opt, h_distance, pop, pop_cost, best_time

def OBMA(A, k, max_n_moves=400, max_iter=50000, max_iter_upd=10000, epsilon=1e-5, t=1500, 
         n_parents=2, ps=10, alpha=15, scale_factor=2.0, timetol=60, is_obl=True, verbose=False):
    """
    This function is an implementation of the Opposition-Based Metaheuristic Algorithm (OBMA) 
    for the maximum weight clique problem.

    The input matrix A represents the weight matrix of the graph, with A[i, j] being the weight
    of the edge between nodes i and j. The input integer k is the number of nodes in the clique. 
    The function has several optional parameters that control the performance and behavior 
    of the algorithm:

    Parameters:
    -----------
        max_n_moves: Maximum number of moves allowed in the tabu search procedure.
        max_iter: Maximum number of iterations allowed in the tabu search procedure.
        max_iter_upd: The convergence criteria for the algorithm, representing the 
            number of consecutive generations with no improvement in the best solution.
        epsilon: A small positive value used as a threshold in the convergence criteria.
        t: The period for determining the tabu tenure.
        n_parents: The number of parents used in the crossover operator.
        ps: The population size.
        alpha: The tabu tenure factor.
        scale_factor: The neighborhood size should not be less than 0.5.
        timetol: The time limit for the algorithm, in seconds.
        is_obl: A boolean value indicating whether to use the opposite solution in the algorithm.
        verbose: A boolean value indicating whether to print verbose output.
    
    Returns:
    --------
    run_vars (dictionary), including multiple keys:
        H: The best solution found by the algorithm.
        obj_score: The weight of the best solution.
        best_time: The time at which the best solution was found.
        local_maximas_h: A list of the solutions found by the algorithm at each 
                         successful update.
        run_trace: A list of tuples, where each tuple contains the weight of the 
                   best solution and the iteration at which it was found.
        params: dictionary containing hyperparameters for the run
    """
    t0 = process_time()
    
    n = A.shape[0]
    no_improve_gen = n_gen = 0
    h_w_opt = -1.0
    best_time = 0.0
    
    h_prime = np.zeros(n, dtype=np.bool_)
    h_opt = np.zeros(n, dtype=np.bool_)
    opposite_h = np.zeros(n, dtype=np.bool_)

    # Initialization
    print("OBMA initialize")
    h_opt, h_w_opt, h_distance, pop, pop_cost, best_time = opposition_based_initialization(
        A, k, n_parents, n, ps, t, t0, max_n_moves, max_iter,
        alpha, scale_factor, timetol, epsilon=epsilon, verbose=verbose
    )
    delta_t1 = process_time() - t0
    print(':: Initialization and first local search completed.')
    print(':: Time elapsed {}, value: {}\n'.format(str(td(seconds=delta_t1)), h_w_opt))
    
    hss = [h_opt]
    run_trace = [(h_w_opt, 0)]
    while True:    
        vprint(f":: h_w_opt = {h_w_opt:.4f}, at gen = {n_gen}, no_improve_gen = {no_improve_gen}"\
              f", time to best = {best_time}.3f\n", verbose)

        # 1. Use the crossover operator
        offspring, opposite_h = crossover_with_greedy(A, pop, n, k, n_parents, ps, is_obl)
        # 2. Run tabu search
        h_w_prime, h_prime, improved_time = tabu_search(
            offspring, A, n, t, t0, max_n_moves, max_iter, 
            alpha, scale_factor, timetol, 
            verbose=verbose
        )
        # 3. Update best solution if necessary
        if h_w_prime > h_w_opt:
            h_opt, h_w_opt, best_time = record_best_solution(
                h_w_prime, h_prime, h_opt, h_w_opt, 
                improved_time, best_time
            )
            print(':: iteration:', n_gen, 'distance in iterations to earlier update:', no_improve_gen)
            print(50*'--')
            hss.append(h_opt)
            run_trace.append((h_w_opt, n_gen))
            no_improve_gen = 0
        else:
            no_improve_gen += 1
            
        # 4. Update population
        pop, pop_cost, h_distance = rank_based_pool_updating(
            h_prime, h_w_prime, alpha, pop, pop_cost, h_distance, ps, k
        )
        if is_obl:
            # 4. Improve opposite solution by tabu search
            h_w_prime, h_prime, improved_time = tabu_search(
                opposite_h, A, n, t, t0, max_n_moves, max_iter, 
                alpha, scale_factor, timetol, verbose=verbose
            )
            # 5. Update best solution if necessary
            if h_w_prime > h_w_opt:
                h_opt, h_w_opt, best_time = record_best_solution(
                    h_w_prime, h_prime, h_opt, h_w_opt, 
                    improved_time, best_time
                )
                print(':: iteration:', n_gen, 'distance in iterations to earlier update:', no_improve_gen)
                print(50*'--')
                hss.append(h_opt)
                run_trace.append((h_w_opt, n_gen))
                no_improve_gen = 0
            else:
                no_improve_gen += 1
            # 6. Update the population
            pop, pop_cost, h_distance = rank_based_pool_updating(
                h_prime, h_w_prime, alpha, pop, pop_cost, h_distance, ps, k
            )
        # 7. Determine stopping / convergence criteria
        if process_time() - t0 >= timetol or no_improve_gen == max_iter_upd:
            break
        n_gen+=1
        
    delta_t = process_time()-t0
    print(':: Run completed @ {} ({} iterations)'.format(str(td(seconds=delta_t)), n_gen))
    print(':: * Best f value: {:.6f} ({:.6f} per node'.format(h_w_opt, h_w_opt / k))
    print(':: * Δt to best f: {}\n'.format(str(td(seconds=best_time))))
    
    local_maximas_h = [np.where(h)[0] for h in hss]
    params = {'k':k,'timetol':timetol,'max_iter':max_iter, 'max_n_moves':max_n_moves, 
              'epsilon':epsilon, 'alpha':alpha, 'scale_factor':scale_factor, 'is_obl':is_obl, 't':t,
              'n_parents':n_parents, 'ps':ps}

    run_vars = {'obj_score':h_w_opt, 'H':np.where(h_opt)[0], 'iterations':n_gen,
                'best_time':best_time, 'init_time':delta_t1, 'local_maximas_h':local_maximas_h,
                'running_time':delta_t, 'run_trace':run_trace, 'pop':pop, 'pop_cost':pop_cost,
                'h_distance':h_distance, 'params':params, 'converged': no_improve_gen >= max_iter_upd}
    return run_vars
