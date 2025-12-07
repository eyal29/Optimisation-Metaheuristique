"""
Fonctions utilitaires pour le main.
Contient toute la logique métier extraite de main.py.
"""

from pareto import generate_fronts, plot_fronts_3d, sol_signature
from algo_utils import gwo_update_population, generate_valid_solution, check_lmax_constraint
from metrics import (
    extract_objectives, init_hv_tracking, update_hv_tracking,
    diversity_spread, spacing_metric, pareto_size,
    load_balancing_index, fog_utilization_ratio, energy_efficiency, average_latency,
    plot_hv_convergence, plot_pareto_2d,
    compute_and_display_metrics, compute_and_display_archive_metrics,
    compute_and_display_alpha_metrics, compute_and_display_archive_solutions_metrics
)


# =============================================================================
# AFFICHAGE DES SOLUTIONS
# =============================================================================

def print_solution_info(solution, idx, lmax_valid, lmax_info):
    """
    Affiche les informations d'une solution de manière formatée.
    
    Args:
        solution: Solution à afficher
        idx: Numéro de la solution (1-based)
        lmax_valid: Booléen indiquant si la contrainte Lmax est respectée
        lmax_info: Dictionnaire avec les informations sur la contrainte Lmax
    """
    print(f"\nSolution {idx}:")
    print(f"  Affectation: {solution.assignment}")
    print(f"  Makespan: {solution.makespan}")
    print(f"  Cout total: {solution.cost}")
    print(f"  Energie totale: {solution.energy}")
    
    if lmax_info is not None:
        status = "✓ Valide" if lmax_valid else "✗ Violée"
        print(f"  Contrainte Lmax: {status} (seuil={lmax_info['threshold']:.2f})")
        if not lmax_valid:
            print(f"    Violations: {lmax_info['violations']}")
    else:
        print("  Contrainte Lmax: N/A (désactivée)")
    
    print("----------")


# =============================================================================
# ÉVALUATION ET FILTRAGE DES SOLUTIONS
# =============================================================================

def evaluate_and_filter_solutions(valid_solutions, donnees, POP_SIZE):
    """
    Évalue les solutions et filtre celles qui respectent la contrainte Lmax.
    
    Args:
        valid_solutions: Liste des solutions à évaluer
        donnees: Données du problème
        POP_SIZE: Taille de la population
    
    Returns:
        Liste des solutions évaluées et valides
    """
    evaluated_solutions = []
    rejected_count = 0
    
    for idx, solution in enumerate(valid_solutions, 1):
        solution.evaluate()
        lmax_valid, lmax_info = check_lmax_constraint(solution, donnees)
        solution.lmax_valid = lmax_valid
        solution.lmax_info = lmax_info
        
        print_solution_info(solution, idx, lmax_valid, lmax_info)
        
        if lmax_valid:
            evaluated_solutions.append(solution)
        else:
            print(" Solution rejetée (ne respecte pas Lmax)")
            rejected_count += 1
    
    # Si aucune solution valide, générer de nouvelles solutions
    if not evaluated_solutions:
        print("ATTENTION: Aucune solution ne respecte Lmax! Génération de nouvelles solutions.")
        evaluated_solutions = [generate_valid_solution(donnees) for _ in range(len(valid_solutions))]
        for sol in evaluated_solutions:
            sol.evaluate()
    
    print(f"\nSolutions acceptées: {len(evaluated_solutions)}/{POP_SIZE}, "
          f"rejetées: {POP_SIZE - len(evaluated_solutions)}")
    
    return evaluated_solutions


def update_and_filter_population(evaluated_solutions, alpha, beta, delta, donnees, a, POP_SIZE):
    """
    Met à jour la population avec GWO et filtre les solutions valides.
    
    Args:
        evaluated_solutions: Population actuelle
        alpha, beta, delta: Leaders de la meute
        donnees: Données du problème
        a: Paramètre de convergence GWO
        POP_SIZE: Taille de la population
    
    Returns:
        Liste des solutions valides après mise à jour
    """
    new_population = gwo_update_population(evaluated_solutions, alpha, beta, delta, donnees, a)
    
    valid_solutions = [sol for sol in new_population 
                       if check_lmax_constraint(sol, donnees)[0]]
    
    # Générer de nouvelles solutions si nécessaire
    max_attempts = 1000
    attempts = 0
    
    while len(valid_solutions) < POP_SIZE and attempts < max_attempts:
        new_sol = generate_valid_solution(donnees)
        if check_lmax_constraint(new_sol, donnees)[0]:
            valid_solutions.append(new_sol)
        attempts += 1
    
    if len(valid_solutions) < POP_SIZE:
        print(f"\n⚠️  ATTENTION: Impossible de trouver {POP_SIZE} solutions valides!")
        print(f"   Seuil Lmax trop restrictif? Trouvé: {len(valid_solutions)}/{POP_SIZE}")
        print(f"   Utilisation de {len(valid_solutions)} solutions pour l'itération suivante.")
    
    return valid_solutions


# =============================================================================
# ARRÊT PRÉCOCE
# =============================================================================

def check_early_stopping(hv_history, prev_hv, iterations_without_improvement, 
                        early_stopping_threshold, early_stopping_patience):
    """
    Vérifie si l'arrêt précoce doit être déclenché basé sur la stagnation de l'hypervolume.
    
    Args:
        hv_history: Historique des hypervolumes
        prev_hv: Hypervolume précédent
        iterations_without_improvement: Nombre d'itérations sans amélioration
        early_stopping_threshold: Seuil d'amélioration minimale
        early_stopping_patience: Nombre d'itérations avant arrêt
    
    Returns:
        Tuple (should_stop, iterations_count, prev_hv)
    """
    if not hv_history:
        return False, iterations_without_improvement, prev_hv
    
    current_hv = hv_history[-1]
    
    if prev_hv is not None:
        improvement = (current_hv - prev_hv) / prev_hv
        
        if improvement < early_stopping_threshold:
            iterations_without_improvement += 1
            print(f"⚠️  HV n'a pas amélioré de {early_stopping_threshold*100}% "
                  f"(amélioration: {improvement*100:.4f}%)")
            print(f"   Itérations sans amélioration: "
                  f"{iterations_without_improvement}/{early_stopping_patience}")
            
            if iterations_without_improvement >= early_stopping_patience:
                print(f"\n ARRÊT PRÉCOCE: Pas d'amélioration depuis "
                      f"{early_stopping_patience} itérations")
                print(f"   HV final: {current_hv:.4f}")
                return True, iterations_without_improvement, current_hv
        else:
            iterations_without_improvement = 0
            print(f"Amélioration détectée: {improvement*100:.4f}%")
    
    return False, iterations_without_improvement, current_hv


# =============================================================================
# AFFICHAGE DES FRONTS ET ARCHIVE
# =============================================================================

def display_fronts(evaluated_solutions, fronts):
    """Affiche les fronts de Pareto de la population actuelle."""
    print("\n===== FRONTS NON DOMINES (population actuelle) =====")
    for i, front in enumerate(fronts, 1):
        print(f"\nFront {i} ({len(front)} solutions) :")
        for solution in front:
            idx = evaluated_solutions.index(solution) + 1 if solution in evaluated_solutions else -1
            print(f"  Solution {idx} :  Makespan={solution.makespan:.4f}, "
                  f"Cost={solution.cost:.4f}, Energy={solution.energy:.4f}")


def display_archive(archive_solutions, show_summary=False, valid_solutions=None, donnees=None):
    """
    Affiche l'archive globale non dominée avec les fronts Pareto.
    
    Args:
        archive_solutions: Solutions de l'archive
        show_summary: Si True, affiche le résumé des statistiques
        valid_solutions: Population finale (requis si show_summary=True)
        donnees: Données du problème (requis si show_summary=True)
    """
    if show_summary and valid_solutions is not None and donnees is not None:
        valid_count_population = sum(1 for sol in valid_solutions 
                                     if check_lmax_constraint(sol, donnees)[0])
        valid_count_archive = sum(1 for sol in archive_solutions 
                                  if hasattr(sol, 'lmax_valid') and sol.lmax_valid)
        
        print("\n" + "="*70)
        print("RÉSUMÉ - Solutions valides vs total")
        print("="*70)
        print(f"Population finale : {valid_count_population}/{len(valid_solutions)} solutions valides")
        print(f"Archive finale : {len(archive_solutions)} solutions (sans doublons)")
        print(f"Solutions valides dans archive (Lmax) : {valid_count_archive}/{len(archive_solutions)}")
        print("="*70)
    
    print("\n===== ARCHIVE (FRONT GLOBAL NON DOMINE) =====")
    
    archive_fronts = generate_fronts(archive_solutions)
    sol_to_idx = {sol_signature(sol): idx + 1 for idx, sol in enumerate(archive_solutions)}
    
    for front_idx, front_solutions in enumerate(archive_fronts, 1):
        print(f"Front {front_idx} ({len(front_solutions)} solutions) :")
        for sol in front_solutions:
            sol_idx = sol_to_idx.get(sol_signature(sol), "?")
            print(f"  Solution {sol_idx} :  Makespan={sol.makespan:.4f}, "
                  f"Cost={sol.cost:.4f}, Energy={sol.energy:.4f}")


def display_leaders(evaluated_solutions, alpha, beta, delta):
    """Affiche les leaders de la population."""
    print("\n===== LEADERS =====")
    for name, solution in [("Alpha", alpha), ("Beta", beta), ("Delta", delta)]:
        if solution is None:
            print(f"{name} : None")
        else:
            idx = evaluated_solutions.index(solution) + 1 if solution in evaluated_solutions else -1
            print(f"{name} = Solution {idx} : "
                  f"Makespan={solution.makespan:.4f}, "
                  f"Cost={solution.cost:.4f}, "
                  f"Energy={solution.energy:.4f}")


# =============================================================================
# RÉSULTATS FINAUX
# =============================================================================

def display_final_summary(valid_solutions, archive_unique, donnees):
    """Affiche le résumé final des solutions valides."""
    display_archive(archive_unique, show_summary=True, 
                   valid_solutions=valid_solutions, donnees=donnees)


def plot_final_results(archive_unique, hv_history, donnees, valid_solutions=None):
    """
    Génère tous les graphiques finaux avec affichage de l'archive complète.
    
    Args:
        archive_unique: Solutions uniques de l'archive
        hv_history: Historique de l'hypervolume
        donnees: Données du problème
        valid_solutions: Solutions valides (optionnel)
    """
    print("\n" + "="*70)
    print("AFFICHAGE DE L'ARCHIVE GLOBALE (TOUS LES FRONTS)")
    print("="*70)
    
    archive_fronts = generate_fronts(archive_unique)
    plot_fronts_3d(archive_unique, archive_fronts, 
                   title="Fronts de Pareto - Archive Globale (Tous les Fronts)")
    print(f"✓ Archive affichée avec {len(archive_fronts)} front(s) "
          f"et {len(archive_unique)} solution(s)")

    if hv_history:
        plot_hv_convergence(hv_history, title="Convergence de l'hypervolume (GWO Fog-Cloud)")

    if archive_unique:
        objs_arch_final = extract_objectives(archive_unique)
        plot_pareto_2d(objs_arch_final, x_idx=0, y_idx=1, 
                      x_label="Makespan", y_label="Cost")

