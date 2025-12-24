"""
MOGWO Standard - Multi-Objective Grey Wolf Optimizer classique
Différences avec l'algorithme hybride :
1. Leaders sélectionnés ALÉATOIREMENT dans l'archive (pas par crowding distance)
2. Archive avec limite de taille (ParetoArchive) pour comparabilité
3. Pas d'early stopping basé sur HV
"""

import numpy as np
from utils_to_algo import gwo_update_population
from algo import compute_a, generate_valid_solution, ParetoArchive


def select_leaders_random(archive):
    """
    Sélection ALÉATOIRE des 3 leaders dans l'archive (MOGWO standard).
    Différence clé : pas de crowding distance, sélection purement aléatoire.
    """
    if not archive or len(archive) == 0:
        return None, None, None
    
    if len(archive) < 3:
        leaders = list(archive) + [archive[0]] * (3 - len(archive))
        return leaders[0], leaders[1], leaders[2]
    
    # Sélection aléatoire
    indices = np.random.choice(len(archive), size=3, replace=False)
    return archive[indices[0]], archive[indices[1]], archive[indices[2]]


def run_mogwo_standard(donnees, config, verbose=True):
    """
    Exécute MOGWO standard et retourne l'historique des hypervolumes.
    Returns:
        hv_history: Liste des hypervolumes à chaque itération
        archive: Archive finale de solutions Pareto
        final_population: Population finale
    """
    from analyses.metrics import init_hv_tracking, update_hv_tracking
    
    # Paramètres
    MAX_ITER = config["gwo"]["max_iter"]
    POP_SIZE = config["gwo"]["population_size"]
    A_MAX = config["gwo"]["a_max"]
    A_MIN = config["gwo"]["a_min"]
    ARCHIVE_MAX = config["gwo"].get("max_archive_size", 100) # Récupérer la taille max
    
    if verbose:
        print("\n" + "="*70)
        print("DÉMARRAGE MOGWO STANDARD")
        print("="*70)
    
    # Initialisation population
    population = [generate_valid_solution(donnees) for _ in range(POP_SIZE)]
    for sol in population:
        sol.evaluate()
    
    # Archive avec limite de taille (ParetoArchive)
    archive = ParetoArchive(max_size=ARCHIVE_MAX)
    for sol in population:
        archive.add(sol)
    
    # Tracking hypervolume
    archive_solutions = archive.get_solutions()
    hv_history, ref_point = init_hv_tracking(archive_solutions)
    
    if verbose:
        print(f"HV initial: {hv_history[0]:.4e}")
    
    # Boucle principale
    for t in range(1, MAX_ITER + 1):
        if verbose and t % 10 == 0:
            print(f"[MOGWO] Itération {t}/{MAX_ITER} - Archive: {len(archive.archive)} solutions - HV: {hv_history[-1]:.4e}")
        
        # Paramètre a
        a = compute_a(t, MAX_ITER, a_max=A_MAX, a_min=A_MIN)
        
        # Sélection ALÉATOIRE des leaders
        archive_list = archive.get_solutions()
        alpha, beta, delta = select_leaders_random(archive_list)
        
        if alpha is None:
            if verbose:
                print(f"⚠️ Pas de leaders disponibles à l'itération {t}")
            break
        
        # Mise à jour population (utilisation de la fonction partagée)
        population = gwo_update_population(population, alpha, beta, delta, donnees, a)
        
        # Mise à jour archive
        for sol in population:
            archive.add(sol)
        
        # Mise à jour hypervolume
        archive_solutions = archive.get_solutions()
        hv_history = update_hv_tracking(archive_solutions, hv_history, ref_point)
    
    if verbose:
        print(f"\n[MOGWO] Terminé - Archive finale: {len(archive.archive)} solutions")
        print(f"HV final: {hv_history[-1]:.4e}")
    
    return hv_history, archive.get_solutions(), population
