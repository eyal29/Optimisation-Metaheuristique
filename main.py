
import io
import sys
import time

from analyses.affichage import compute_and_display_metrics, display_archive, display_fronts, display_leaders, finalize_and_report
from algo_hybride.algo import check_early_stopping, compute_a, evaluate_solutions, generate_fronts, select_leaders
from algo_hybride.initialization import  initialize_full_algorithm
from algo_hybride.utils_to_algo import gwo_update_population
from analyses.visualization import plot_final_results


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def main():
    start_time = time.time()
    
    # 1. INITIALISATION ET SETUP
    donnees, config, valid_solutions, archive, hv_history, ref_point, iterations_without_improvement, prev_hv = \
        initialize_full_algorithm()
    
    # Paramètres de l'exécution
    MAX_ITER = config["gwo"]["max_iter"]
    A_MAX = config["gwo"]["a_max"]
    A_MIN = config["gwo"]["a_min"]
    EARLY_STOPPING_THRESHOLD = config["gwo"]["early_stopping_threshold"]
    EARLY_STOPPING_PATIENCE = config["gwo"]["early_stopping_patience"]
    POP_SIZE = config["gwo"]["population_size"]

    # 2. BOUCLE PRINCIPALE D'EXÉCUTION (GWO-NSGA-II)
    for t in range(1, MAX_ITER + 1):
        print(f"\n===== ITERATION {t} =====")
        a = compute_a(t, MAX_ITER, a_max=A_MAX, a_min=A_MIN)
        print(f"a = {a:.4f}")

        # Évaluation, Fronts, Archive
        evaluated_solutions = evaluate_solutions(valid_solutions, donnees, POP_SIZE)
        fronts = generate_fronts(evaluated_solutions)
        display_fronts(evaluated_solutions, fronts)
        for sol in evaluated_solutions:
            archive.add(sol)

        # Métriques et Arrêt Précoce
        archive_solutions = archive.get_solutions()
        display_archive(archive_solutions)
        print("\n===== MÉTRIQUES ARCHIVE =====")
        hv_history, ref_point = compute_and_display_metrics(archive_solutions, 'archive', 
                                      hv_history=hv_history, ref_point=ref_point)
        
        should_stop, iterations_without_improvement, prev_hv = check_early_stopping(
            hv_history, prev_hv, iterations_without_improvement,
            EARLY_STOPPING_THRESHOLD, EARLY_STOPPING_PATIENCE
        )
        if should_stop:
            break

        # Mise à jour GWO
        alpha, beta, delta = select_leaders(evaluated_solutions)
        display_leaders(evaluated_solutions, alpha, beta, delta)
        compute_and_display_metrics(alpha, 'solution', donnees=donnees)
        
        valid_solutions = gwo_update_population(
            evaluated_solutions, alpha, beta, delta, donnees, a
        )

    # 3. FINALISATION ET RAPPORT
    finalize_and_report(archive, donnees, valid_solutions, hv_history, start_time, plot_final_results)


if __name__ == "__main__":
    main()