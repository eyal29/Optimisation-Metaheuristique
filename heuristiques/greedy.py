import numpy as np
from algo_hybride.utils_to_algo import Solution 

def generate_greedy_solution(donnees, greedy_mode='makespan'):
    """
    Génère une solution gloutonne mono-objectif tout en respectant la contrainte mémoire.
    """
    assignment = np.zeros(donnees.n, dtype=int)
    memory_used = np.zeros(donnees.p, dtype=float)
    
    def compute_local_cost(i, j):
        U = donnees.U_ij[i, j]
        m = donnees.m_i[i]
        q = donnees.q_i[i]
        
        if greedy_mode == 'makespan':
            return U
        elif greedy_mode == 'cost':
            return (donnees.lambda_j[j] * U + donnees.beta_j[j] * m + donnees.gamma_j[j] * q)
        elif greedy_mode == 'energy':
            return U * donnees.energy_j[j]
        return np.inf

    for i in range(donnees.n):
        best_vm = -1
        min_cost = np.inf
        
        for j in range(donnees.p):
            if memory_used[j] + donnees.m_i[i] <= donnees.memory_capacity[j]:
                current_cost = compute_local_cost(i, j)
                
                if current_cost < min_cost:
                    min_cost = current_cost
                    best_vm = j
        
        if best_vm != -1:
            assignment[i] = best_vm
            memory_used[best_vm] += donnees.m_i[i]
        else:
            pass 
            
    solution = Solution(assignment.astype(int), donnees)
    solution.evaluate()
    return solution