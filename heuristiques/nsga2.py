import copy
import numpy as np
from algo import generate_fronts, generate_valid_solution
from utils_to_algo import assign_crowding_distance, Solution
from utils_to_algo import repair_assignment 
# 💡 IMPORTS POUR L'HYPERVOLUME
from analyses.metrics import init_hv_tracking, update_hv_tracking, extract_objectives


def simple_mutation(solution, donnees, mutation_rate=0.1):
    """
    Applique une simple mutation à l'affectation d'une solution.
    """
    new_assignment = solution.assignment.copy()
    n, p = donnees.n, donnees.p

    for i in range(n):
        if np.random.rand() < mutation_rate:
            # Réassignation aléatoire
            new_assignment[i] = np.random.randint(0, p)

    repaired_assignment = repair_assignment(new_assignment.astype(int), donnees)
    
    new_sol = Solution(repaired_assignment, donnees)
    return new_sol

def solve_nsga2_simple(donnees, config):
    """
    Résout le problème avec une approche NSGA-II simple.
    Retourne le Front de Pareto final (F1) et l'historique de l'Hypervolume (HV).
    """
    POP_SIZE = config["gwo"]["population_size"]
    MAX_ITER = config["gwo"]["max_iter"]
    
    population = [generate_valid_solution(donnees) for _ in range(POP_SIZE)]
    for sol in population:
        sol.evaluate()
        
    hv_history = None
    ref_point = None
    
    # 💡 Initialisation de l'HV
    # On utilise la population initiale pour définir le point de référence
    hv_history, ref_point = init_hv_tracking(population)

    for t in range(1, MAX_ITER + 1):
        
        # 1. Sélection (Tri Pareto + Crowding)
        fronts_idx = generate_fronts(population, return_indices=True)
        
        selected_indices = []
        for front in fronts_idx:
            if len(selected_indices) + len(front) <= POP_SIZE:
                selected_indices.extend(front)
            else:
                remaining = POP_SIZE - len(selected_indices)
                assign_crowding_distance(population, [front])
                front_sorted = sorted(front, key=lambda i: population[i].crowding_distance, reverse=True)
                selected_indices.extend(front_sorted[:remaining])
                break 
        
        parents = [population[i] for i in selected_indices]
        
        # 💡 Mise à jour de l'HV à partir de la nouvelle population (parents sélectionnés)
        hv_history = update_hv_tracking(parents, hv_history, ref_point)

        # 2. Reproduction (Simple Mutation)
        new_population = []
        for sol in parents:
            new_sol = simple_mutation(sol, donnees, mutation_rate=0.1) 
            new_sol.evaluate()
            new_population.append(new_sol)

        population = new_population

    # Retourner le Front 1 et l'historique HV
    fronts_final = generate_fronts(population)
    final_front = fronts_final[0] if fronts_final else []
    
    return final_front, hv_history # 💡 Retourne les deux éléments
