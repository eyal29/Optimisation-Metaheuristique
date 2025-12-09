# Fichier: gwo_mono_solver.py

import copy
from algo import compute_a, generate_valid_solution
from utils_to_algo import gwo_update_population


def select_mono_leaders(population, objective_mode):
    
    if not population:
        return None, None, None
        
    # Choisir la métrique à utiliser pour le tri
    if objective_mode == 'makespan':
        key_func = lambda s: s.makespan
    elif objective_mode == 'cost':
        key_func = lambda s: s.cost
    elif objective_mode == 'energy':
        key_func = lambda s: s.energy
    else:
        raise ValueError("Mode d'objectif invalide pour GWO mono-objectif.")

    sorted_population = sorted(population, key=key_func)
    
    alpha = sorted_population[0] if len(sorted_population) >= 1 else None
    beta  = sorted_population[1] if len(sorted_population) >= 2 else alpha
    delta = sorted_population[2] if len(sorted_population) >= 3 else beta
    
    if alpha is beta: beta = copy.deepcopy(alpha)
    if alpha is delta: delta = copy.deepcopy(alpha)

    return alpha, beta, delta


def solve_gwo_mono(donnees, config, objective_mode):
   
    POP_SIZE = config["gwo"]["population_size"]
    MAX_ITER = config["gwo"]["max_iter"]
    A_MAX = config["gwo"]["a_max"]
    A_MIN = config["gwo"]["a_min"]
    
    population = [generate_valid_solution(donnees) for _ in range(POP_SIZE)]
    best_solution = None

    for t in range(1, MAX_ITER + 1):
        a = compute_a(t, MAX_ITER, a_max=A_MAX, a_min=A_MIN)
        
        for sol in population:
            sol.evaluate()
            if best_solution is None or select_mono_leaders([sol, best_solution], objective_mode)[0] is sol:
                 best_solution = copy.deepcopy(sol)
        
        alpha, beta, delta = select_mono_leaders(population, objective_mode)
        
        population = gwo_update_population(population, alpha, beta, delta, donnees, a)
        
    return best_solution