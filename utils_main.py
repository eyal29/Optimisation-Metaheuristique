"""
Fonctions utilitaires pour le main.
Contient toute la logique métier extraite de main.py
"""

from solutions_definiton import Donnees
from pareto import generate_fronts, select_leaders, plot_fronts_3d, ParetoArchive
from algo_utils import gwo_update_population, generate_valid_solution, compute_a, check_lmax_constraint
from metrics import *


def evaluate_and_filter_solutions(valid_solutions, donnees, POP_SIZE):
    """Évalue les solutions et filtre celles qui respectent la contrainte Lmax."""
    evaluated_solutions = []
    rejected_count = 0
    
    for idx, solution in enumerate(valid_solutions):
        solution.evaluate()
        
        # Vérifier la contrainte Lmax
        lmax_valid, lmax_info = check_lmax_constraint(solution, donnees)
        solution.lmax_valid = lmax_valid
        solution.lmax_info = lmax_info
        
        print(f"\nSolution {idx + 1}:")
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


def display_fronts(evaluated_solutions, fronts):
    """Affiche les fronts de Pareto de la population actuelle."""
    print("\n===== FRONTS NON DOMINES (population actuelle) =====")
    for i, front in enumerate(fronts):
        print(f"\nFront {i+1} :")
        for solution in front:
            try:
                idx = evaluated_solutions.index(solution) + 1
            except ValueError:
                idx = -1
            print(f"  Solution {idx} :  Makespan={solution.makespan:.4f}, "
                  f"Cost={solution.cost:.4f}, Energy={solution.energy:.4f}")


def display_archive(archive_solutions):
    """Affiche l'archive globale non dominée."""
    print("\n===== ARCHIVE (FRONT GLOBAL NON DOMINE) =====")
    for sol in archive_solutions:
        print(f"  Makespan={sol.makespan:.4f}, "
            f"Cost={sol.cost:.4f}, Energy={sol.energy:.4f}")


def compute_and_display_archive_metrics(archive_solutions, hv_history, ref_point):
    """Calcule et affiche les métriques de l'archive."""
    if len(archive_solutions) > 0:
        objs_arch = extract_objectives(archive_solutions)
        if hv_history is None:
            hv_history, ref_point = init_hv_tracking(archive_solutions)
        else:
            hv_history = update_hv_tracking(archive_solutions, hv_history, ref_point)

        div = diversity_spread(objs_arch)
        sp = spacing_metric(objs_arch)
        psize = pareto_size(objs_arch)
        print(f"\n[Metrics ARCHIVE] HV={hv_history[-1]:.4f}, "
            f"Diversity={div:.4f}, Spacing={sp:.4f}, "
            f"Pareto size={psize}")
    
    return hv_history, ref_point


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


def compute_and_display_alpha_metrics(alpha, donnees):
    """Calcule et affiche les métriques du leader alpha."""
    if alpha is not None:
        lbi = load_balancing_index(alpha, donnees)
        fur = fog_utilization_ratio(alpha, donnees)
        ee = energy_efficiency(alpha)
        avg_lat = average_latency(alpha, donnees)
        print(f"\n[Metrics ALPHA] LBI={lbi:.4f}, FUR={fur:.4f}, "
              f"EE={ee:.4f}, AvgLatency={avg_lat:.4f}")


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
    valid_count_population = 0
    for sol in valid_solutions:
        lmax_valid, _ = check_lmax_constraint(sol, donnees)
        if lmax_valid:
            valid_count_population += 1

    print("\n" + "="*70)
    print(f"RÉSUMÉ - Solutions valides vs total")
    print("="*70)
    print(f"Population finale : {valid_count_population}/{len(valid_solutions)} solutions valides")
    print(f"Archive finale : {len(archive_unique)} solutions (sans doublons)")
    
    valid_count_archive = sum(1 for sol in archive_unique if hasattr(sol, 'lmax_valid') and sol.lmax_valid)
    print(f"Solutions valides dans archive (Lmax) : {valid_count_archive}/{len(archive_unique)}")
    print("="*70)

    print("\n===== MEILLEURES SOLUTIONS (ARCHIVE FINALE SANS DOUBLONS) =====")
    for idx, sol in enumerate(archive_unique, 1):
        print(f"  Solution {idx}: Makespan={sol.makespan:.4f}, "
              f"Cost={sol.cost:.4f}, Energy={sol.energy:.4f}")


def plot_final_results(archive_unique, hv_history, donnees):
    """Génère tous les graphiques finaux."""
    archive_fronts = generate_fronts(archive_unique)
    plot_fronts_3d(archive_unique, archive_fronts)

    if hv_history is not None:
        plot_hv_convergence(hv_history, title="Convergence de l'hypervolume (GWO Fog-Cloud)")

    if len(archive_unique) > 0:
        best = min(archive_unique, key=lambda s: s.makespan)

        metrics_dict = {
            "LBI": load_balancing_index(best, donnees),
            "FUR": fog_utilization_ratio(best, donnees),
            "EE": energy_efficiency(best),
            "Avg Latency": average_latency(best, donnees),
        }
        plot_metrics_subplots(metrics_dict, title="Métriques Fog-Cloud (meilleure solution archive)")

        objs_arch_final = extract_objectives(archive_unique)
        plot_pareto_2d(
            objs_arch_final,
            x_idx=0, y_idx=1,
            x_label="Makespan", y_label="Cost"
        )
