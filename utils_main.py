"""
Fonctions utilitaires pour le main.
Contient toute la logique métier extraite de main.py
"""

# Algorithmes et Pareto
from pareto import generate_fronts, plot_fronts_3d, sol_signature
from algo_utils import gwo_update_population, generate_valid_solution, check_lmax_constraint

# Métriques (import * nécessaire pour toutes les fonctions de métriques)
from metrics import *


def print_solution_info(solution, idx, lmax_valid, lmax_info):
    """Affiche les informations d'une solution de manière formatée.
    
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
        print(f"  Contrainte Lmax: N/A (désactivée)")
    
    print("----------")


def evaluate_and_filter_solutions(valid_solutions, donnees, POP_SIZE):
    """Évalue les solutions et filtre celles qui respectent la contrainte Lmax."""
    evaluated_solutions = []
    rejected_count = 0
    
    for idx, solution in enumerate(valid_solutions, 1):
        solution.evaluate()
        
        # Vérifier la contrainte Lmax
        lmax_valid, lmax_info = check_lmax_constraint(solution, donnees)
        solution.lmax_valid = lmax_valid
        solution.lmax_info = lmax_info
        
        # Afficher les informations de la solution
        print_solution_info(solution, idx, lmax_valid, lmax_info)
        
        # Garder seulement les solutions valides
        if lmax_valid:
            evaluated_solutions.append(solution)
        else:
            print(f" Solution rejetée (ne respecte pas Lmax)")
            rejected_count += 1
    
    # Si aucune solution valide, générer de nouvelles solutions
    if not evaluated_solutions:
        print("ATTENTION: Aucune solution ne respecte Lmax! Génération de nouvelles solutions.")
        evaluated_solutions = [generate_valid_solution(donnees) for _ in range(len(valid_solutions))]
        for sol in evaluated_solutions:
            sol.evaluate()
    
    print(f"\nSolutions acceptées: {len(evaluated_solutions)}/{POP_SIZE}, rejetées: {POP_SIZE - len(evaluated_solutions)}")
    
    return evaluated_solutions


def check_early_stopping(hv_history, prev_hv, iterations_without_improvement, 
                        early_stopping_threshold, early_stopping_patience):
    """Vérifie si l'arrêt précoce doit être déclenché basé sur la stagnation de l'hypervolume.
    
    Args:
        hv_history: Historique des hypervolumes
        prev_hv: Hypervolume précédent
        iterations_without_improvement: Nombre d'itérations sans amélioration
        early_stopping_threshold: Seuil d'amélioration minimale (chargé depuis config.yaml)
        early_stopping_patience: Nombre d'itérations avant arrêt (chargé depuis config.yaml)
    
    Returns:
        Tuple (should_stop, iterations_count, prev_hv): 
            - should_stop: Booléen indiquant si on doit arrêter
            - iterations_count: Compteur mis à jour
            - prev_hv: Nouvelle valeur de prev_hv
    """
    if hv_history is None or len(hv_history) == 0:
        return False, iterations_without_improvement, prev_hv
    
    current_hv = hv_history[-1]
    
    if prev_hv is not None:
        # Calculer le pourcentage d'amélioration
        improvement = (current_hv - prev_hv) / prev_hv
        
        if improvement < early_stopping_threshold:
            iterations_without_improvement += 1
            print(f"⚠️  HV n'a pas amélioré de {early_stopping_threshold*100}% (amélioration: {improvement*100:.4f}%)")
            print(f"   Itérations sans amélioration: {iterations_without_improvement}/{early_stopping_patience}")
            
            # Arrêt si pas d'amélioration pendant trop longtemps
            if iterations_without_improvement >= early_stopping_patience:
                print(f"\n ARRÊT PRÉCOCE: Pas d'amélioration depuis {early_stopping_patience} itérations")
                print(f"   HV final: {current_hv:.4f}")
                return True, iterations_without_improvement, current_hv
        else:
            # Réinitialiser le compteur si amélioration détectée
            iterations_without_improvement = 0
            print(f"Amélioration détectée: {improvement*100:.4f}%")
    
    return False, iterations_without_improvement, current_hv


def display_fronts(evaluated_solutions, fronts):
    """Affiche les fronts de Pareto de la population actuelle."""
    print("\n===== FRONTS NON DOMINES (population actuelle) =====")
    for i, front in enumerate(fronts):
        print(f"\nFront {i+1} ({len(front)} solutions) :")
        for solution in front:
            try:
                idx = evaluated_solutions.index(solution) + 1
            except ValueError:
                idx = -1
            print(f"  Solution {idx} :  Makespan={solution.makespan:.4f}, "
                  f"Cost={solution.cost:.4f}, Energy={solution.energy:.4f}")


def display_archive(archive_solutions, show_summary=False, valid_solutions=None, donnees=None):
    """Affiche l'archive globale non dominée avec les fronts Pareto.
    
    Args:
        archive_solutions: Solutions de l'archive
        show_summary: Si True, affiche le résumé des statistiques
        valid_solutions: Population finale (requis si show_summary=True)
        donnees: Données du problème (requis si show_summary=True)
    """
    # Afficher le résumé si demandé
    if show_summary and valid_solutions is not None and donnees is not None:
        valid_count_population = sum(1 for sol in valid_solutions if check_lmax_constraint(sol, donnees)[0])
        valid_count_archive = sum(1 for sol in archive_solutions if hasattr(sol, 'lmax_valid') and sol.lmax_valid)
        
        print("\n" + "="*70)
        print(f"RÉSUMÉ - Solutions valides vs total")
        print("="*70)
        print(f"Population finale : {valid_count_population}/{len(valid_solutions)} solutions valides")
        print(f"Archive finale : {len(archive_solutions)} solutions (sans doublons)")
        print(f"Solutions valides dans archive (Lmax) : {valid_count_archive}/{len(archive_solutions)}")
        print("="*70)
    
    print("\n===== ARCHIVE (FRONT GLOBAL NON DOMINE) =====")
    
    # Générer les fronts de Pareto pour l'archive
    archive_fronts = generate_fronts(archive_solutions)
    
    # Créer un mapping de solution vers son index dans archive_solutions
    sol_to_idx = {sol_signature(sol): idx + 1 for idx, sol in enumerate(archive_solutions)}
    
    # Afficher l'archive groupée par fronts
    for front_idx, front_solutions in enumerate(archive_fronts, 1):
        print(f"Front {front_idx} ({len(front_solutions)} solutions) :")
        for sol in front_solutions:
            sol_idx = sol_to_idx.get(sol_signature(sol), "?")
            print(f"  Solution {sol_idx} :  Makespan={sol.makespan:.4f}, "
                  f"Cost={sol.cost:.4f}, Energy={sol.energy:.4f}")


def compute_and_display_metrics(target, metric_type, donnees=None, hv_history=None, ref_point=None):
    """Calcule et affiche les métriques selon le type demandé.
    
    Args:
        target: Archive (list) ou solution individuelle
        metric_type: 'archive' ou 'solution'
        donnees: Données du problème (requis pour metric_type='solution')
        hv_history: Historique hypervolume (pour metric_type='archive')
        ref_point: Point de référence (pour metric_type='archive')
        
    Returns:
        tuple: (hv_history, ref_point) si metric_type='archive', sinon None
    """
    if metric_type == 'archive' and len(target) > 0:
        # Métriques de l'archive
        objs_arch = extract_objectives(target)
        if hv_history is None:
            hv_history, ref_point = init_hv_tracking(target)
        else:
            hv_history = update_hv_tracking(target, hv_history, ref_point)

        div = diversity_spread(objs_arch)
        sp = spacing_metric(objs_arch)
        psize = pareto_size(objs_arch)
        print(f"\n[Metrics ARCHIVE] HV={hv_history[-1]:.4f}, "
              f"Diversity={div:.4f}, Spacing={sp:.4f}, "
              f"Pareto size={psize}")
        
        return hv_history, ref_point
    
    elif metric_type == 'solution' and target is not None and donnees is not None:
        # Métriques d'une solution individuelle
        lbi = load_balancing_index(target, donnees)
        fur = fog_utilization_ratio(target, donnees)
        ee = energy_efficiency(target)
        avg_lat = average_latency(target, donnees)
        print(f"\n[Metrics SOLUTION] LBI={lbi:.4f}, FUR={fur:.4f}, "
              f"EE={ee:.4f}, AvgLatency={avg_lat:.4f}")
        
        return None
    
    # Cas par défaut pour metric_type='archive'
    return hv_history, ref_point


def compute_and_display_archive_metrics(archive_solutions, hv_history, ref_point):
    """Calcule et affiche les métriques de l'archive (wrapper de compatibilité)."""
    return compute_and_display_metrics(archive_solutions, 'archive', hv_history=hv_history, ref_point=ref_point)


def compute_and_display_alpha_metrics(alpha, donnees):
    """Calcule et affiche les métriques du leader alpha (wrapper de compatibilité)."""
    compute_and_display_metrics(alpha, 'solution', donnees=donnees)


def compute_and_display_archive_solutions_metrics(archive_solutions, donnees):
    """
    Calcule et affiche les métriques pour TOUTES les solutions de l'archive.
    
    Args:
        archive_solutions : liste des solutions de l'archive
        donnees : Données du problème
    """
    print("\n" + "="*70)
    print("MÉTRIQUES FINALES - TOUTES LES SOLUTIONS DE L'ARCHIVE")
    print("="*70)
    
    if len(archive_solutions) > 0:
        for idx, solution in enumerate(archive_solutions, 1):
            print(f"\n--- Solution {idx} ---")
            compute_and_display_metrics(solution, 'solution', donnees=donnees)
    else:
        print("Aucune solution dans l'archive")


def display_leaders(evaluated_solutions, alpha, beta, delta):
    """Affiche les leaders de la population."""
    print("\n===== LEADERS =====")
    for name, solution in [("Alpha", alpha), ("Beta", beta), ("Delta", delta)]:
        if solution is None:
            print(f"{name} : None")
        else:
            try:
                idx = evaluated_solutions.index(solution) + 1
            except ValueError:
                idx = -1
            print(f"{name} = Solution {idx} : "
                  f"Makespan={solution.makespan:.4f}, "
                  f"Cost={solution.cost:.4f}, "
                  f"Energy={solution.energy:.4f}")


def update_and_filter_population(evaluated_solutions, alpha, beta, delta, donnees, a, POP_SIZE):
    """Met à jour la population avec GWO et filtre les solutions valides."""
    new_population = gwo_update_population(
        evaluated_solutions, alpha, beta, delta, donnees, a
    )
    
    # Filtrer les nouvelles solutions pour garder seulement les valides
    valid_solutions = []
    max_attempts = 1000
    attempts = 0
    
    for sol in new_population:
        lmax_valid, _ = check_lmax_constraint(sol, donnees)
        if lmax_valid:
            valid_solutions.append(sol)
    
    # Si pas assez de solutions valides, en générer de nouvelles (avec limite)
    while len(valid_solutions) < POP_SIZE and attempts < max_attempts:
        new_sol = generate_valid_solution(donnees)
        lmax_valid, _ = check_lmax_constraint(new_sol, donnees)
        if lmax_valid:
            valid_solutions.append(new_sol)
        attempts += 1
    
    # Si on n'a pas assez de solutions valides après max_attempts
    if len(valid_solutions) < POP_SIZE:
        print(f"\n⚠️  ATTENTION: Impossible de trouver {POP_SIZE} solutions valides!")
        print(f"   Seuil Lmax trop restrictif? Trouvé: {len(valid_solutions)}/{POP_SIZE}")
        print(f"   Utilisation de {len(valid_solutions)} solutions valides pour l'itération suivante.")
    
    return valid_solutions


def display_final_summary(valid_solutions, archive_unique, donnees):
    """Affiche le résumé final des solutions valides."""
    display_archive(archive_unique, show_summary=True, valid_solutions=valid_solutions, donnees=donnees)


def plot_final_results(archive_unique, hv_history, donnees, valid_solutions=None):
    """Génère tous les graphiques finaux avec affichage de l'archive complète."""
    
    # Afficher l'archive avec tous les fronts distingués
    print("\n" + "="*70)
    print("AFFICHAGE DE L'ARCHIVE GLOBALE (TOUS LES FRONTS)")
    print("="*70)
    archive_fronts = generate_fronts(archive_unique)
    plot_fronts_3d(archive_unique, archive_fronts, title="Fronts de Pareto - Archive Globale (Tous les Fronts)")
    print(f"✓ Archive affichée avec {len(archive_fronts)} front(s) et {len(archive_unique)} solution(s)")

    # Autres graphiques
    if hv_history is not None:
        plot_hv_convergence(hv_history, title="Convergence de l'hypervolume (GWO Fog-Cloud)")

    if len(archive_unique) > 0:
        objs_arch_final = extract_objectives(archive_unique)
        plot_pareto_2d(
            objs_arch_final,
            x_idx=0, y_idx=1,
            x_label="Makespan", y_label="Cost"
        )
