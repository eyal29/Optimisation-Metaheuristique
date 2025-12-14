import time
from algo import generate_fronts
from heuristiques.greedy import generate_greedy_solution
from heuristiques.gwo_simple import solve_gwo_mono
from analyses.metrics import average_cost_per_video, average_latency, compute_metrics_all_solutions, diversity_spread, energy_efficiency, extract_objectives, fog_utilization_ratio, init_hv_tracking, load_balancing_index, pareto_size, spacing_metric, update_hv_tracking
from utils_to_algo import sol_signature


def print_solution_info(solution, idx,):
    print(f"\nSolution {idx}:")
    print(f"  Affectation: {solution.assignment}")
    print("----------")


def display_fronts(evaluated_solutions, fronts):
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
    """
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


def compute_and_display_metrics(target, metric_type, donnees=None, hv_history=None, ref_point=None):
    """
    Calcule et affiche les métriques selon le type demandé.
    Args:
        target: Archive (list) ou solution individuelle
        metric_type: 'archive' ou 'solution'
        donnees: Données du problème (requis pour metric_type='solution')
        hv_history: Historique hypervolume (pour metric_type='archive')
        ref_point: Point de référence (pour metric_type='archive')
    """
    if metric_type == 'archive' and target:
        objs_arch = extract_objectives(target)
        
        if hv_history is None:
            hv_history, ref_point = init_hv_tracking(target)
        else:
            hv_history = update_hv_tracking(target, hv_history, ref_point)

        div = diversity_spread(objs_arch)
        sp = spacing_metric(objs_arch)
        psize = pareto_size(objs_arch)
        
        print(f"\n[Metrics ARCHIVE] HV={hv_history[-1]:.4f}, "
              f"Diversity={div:.4f}, Spacing={sp:.4f}, Pareto size={psize}")
        
        return hv_history, ref_point
    
    elif metric_type == 'solution' and target is not None and donnees is not None:
        lbi = load_balancing_index(target, donnees)
        fur = fog_utilization_ratio(target, donnees)
        ee = energy_efficiency(target)
        avg_lat = average_latency(target, donnees)
        
        print(f"\n[Metrics SOLUTION] LBI={lbi:.4f}, FUR={fur:.4f}, "
              f"EE={ee:.4f}, AvgLatency={avg_lat:.4f}")
        
        return None
    
    return hv_history, ref_point


def compute_and_display_archive_solutions_metrics(archive_solutions, donnees):
    """
    Calcule et affiche les métriques pour toutes les solutions de l'archive.
    """
    print("\n" + "="*70)
    print("MÉTRIQUES FINALES - TOUTES LES SOLUTIONS DE L'ARCHIVE")
    print("="*70)
    
    if archive_solutions:
        for idx, solution in enumerate(archive_solutions, 1):
            print(f"\n--- Solution {idx} ---")
            compute_and_display_metrics(solution, 'solution', donnees=donnees)
    else:
        print("Aucune solution dans l'archive")


def finalize_and_report(archive, donnees, valid_solutions, hv_history, start_time, plot_final_results):
    # Nettoyage de l'archive
    archive_unique = archive.get_solutions()
    
    print("\n" + "="*70)
    print("COMPARAISON GWO-NSGA-II vs. AUTRES ALGORITHMES")
    print("="*70)
    
    print("\n[Greedy] Génération des 3 solutions mono-objectives...")
    
    sol_greedy_m = generate_greedy_solution(donnees, greedy_mode='makespan')
    sol_greedy_c = generate_greedy_solution(donnees, greedy_mode='cost')
    sol_greedy_e = generate_greedy_solution(donnees, greedy_mode='energy')

    print("\n[GWO Mono] Génération des 3 solutions mono-objectives...")
    from initialization import load_config # Temporaire, pour le test
    config = load_config("config.yaml")
    sol_gwo_m = solve_gwo_mono(donnees, config, objective_mode='makespan')
    sol_gwo_c = solve_gwo_mono(donnees, config, objective_mode='cost')
    sol_gwo_e = solve_gwo_mono(donnees, config, objective_mode='energy')

    reference_solutions = {
        'Greedy-Makespan': sol_greedy_m,
        'Greedy-Cost': sol_greedy_c,
        'Greedy-Energy': sol_greedy_e,
        'GWO-Mono-Makespan': sol_gwo_m,
        'GWO-Mono-Cost': sol_gwo_c,
        'GWO-Mono-Energy': sol_gwo_e
    }
    print("\n--- SOLUTIONS DE RÉFÉRENCE ---")
    for name, sol in reference_solutions.items():
        print(f"[{name}] Makespan={sol.makespan:.4f}, Cost={sol.cost:.4f}, Energy={sol.energy:.4f}")

    # RAPPORT FINAL (AFFICHAGE ET VISUALISATION)
    print("\n\n===== DÉTAILS DES SOLUTIONS FINALES DE L'ARCHIVE =====")
    for idx, solution in enumerate(archive_unique, 1):
        print_solution_info(solution, idx)
    display_archive(archive_unique, show_summary=True, valid_solutions=valid_solutions, donnees=donnees)
    metrics_data = compute_metrics_all_solutions(archive_unique, donnees)

    end_time = time.time()
    exec_time = end_time - start_time
    print(f"\nTemps d'exécution total : {exec_time:.2f} secondes")

    plot_final_results(archive_unique, hv_history, donnees, valid_solutions, reference_solutions, metrics_data)