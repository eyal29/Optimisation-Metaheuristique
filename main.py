
 
# Redirection de la sortie standard vers un flux avec encodage UTF-8
import io
import sys
import time

from affichage import compute_and_display_metrics, display_archive, display_fronts, display_leaders, print_solution_info
from algo import ParetoArchive, check_early_stopping, compute_a, evaluate_and_filter_solutions, generate_fronts, select_leaders
from metrics import compute_metrics_all_solutions
from initialization import initialize_algorithm
from utils_to_algo import gwo_update_population
from visualization import plot_archive_metrics_visualization, plot_final_results


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
  # Initialisation des paramètres et des données
    donnees, config, valid_solutions = initialize_algorithm("config.yaml")

    # Paramètres du GWO extraits de la configuration
    MAX_ITER = config["gwo"]["max_iter"]
    POP_SIZE = config["gwo"]["population_size"]
    ARCHIVE_MAX = config["gwo"]["max_archive_size"]
    A_MAX = config["gwo"]["a_max"]
    A_MIN = config["gwo"]["a_min"]
    EARLY_STOPPING_THRESHOLD = config["gwo"]["early_stopping_threshold"]
    EARLY_STOPPING_PATIENCE = config["gwo"]["early_stopping_patience"]

    # Initialisation de l'archive
    archive = ParetoArchive(max_size=ARCHIVE_MAX)
    
   
    hv_history = None
    ref_point = None


    start_time = time.time()

    # Paramètres d'arrêt précoce
    iterations_without_improvement = 0
    prev_hv = None
    
    # Historiques des métriques pour le leader alpha
    metrics_history = {
        'LBI': [],
        'FUR': [],
        'EE': [],
        'AvgLatency': []
    }

    # BOUCLE PRINCIPALE
    for t in range(1, MAX_ITER + 1):
        print(f"\n===== ITERATION {t} =====")
        a = compute_a(t, MAX_ITER, a_max=A_MAX, a_min=A_MIN)
        print(f"a = {a:.4f}")

        # Évaluation 
        evaluated_solutions = evaluate_and_filter_solutions(valid_solutions, donnees, POP_SIZE)

        # Fronts de la population actuelle
        fronts = generate_fronts(evaluated_solutions)
        display_fronts(evaluated_solutions, fronts)

        # Ajout à l'archive
        for sol in evaluated_solutions:
            archive.add(sol)

        # Affichage de l'archive
        archive_solutions = archive.get_solutions()
        display_archive(archive_solutions)

        # AFFICHAGE CLAIR des METRICS ARCHIVE
        print("\n===== MÉTRIQUES ARCHIVE =====")

        # Métriques sur l'archive
        hv_history, ref_point = compute_and_display_metrics(archive_solutions, 'archive', 
                                      hv_history=hv_history, ref_point=ref_point)

        # Vérifier l'arrêt précoce basé sur l'hypervolume
        should_stop, iterations_without_improvement, prev_hv = check_early_stopping(
            hv_history, prev_hv, iterations_without_improvement,
            EARLY_STOPPING_THRESHOLD, EARLY_STOPPING_PATIENCE
        )
        if should_stop:
            break

        # Sélection des leaders
        alpha, beta, delta = select_leaders(evaluated_solutions)
        display_leaders(evaluated_solutions, alpha, beta, delta)

        # Métriques sur le leader alpha
        compute_and_display_metrics(alpha, 'solution', donnees=donnees)

        # Mise à jour de la population (GWO)
        valid_solutions = gwo_update_population(
            evaluated_solutions, alpha, beta, delta, donnees, a
        )

    # Nettoyage de l'archive
    archive_unique = archive.get_solutions()

    print("\n\n===== DÉTAILS DES SOLUTIONS FINALES DE L'ARCHIVE =====")
    for idx, solution in enumerate(archive_unique, 1):
        print_solution_info(solution, idx)
    # Résumé final
    display_archive(archive_unique, show_summary=True, valid_solutions=valid_solutions, donnees=donnees)
    
    # Calcul des métriques pour TOUTES les solutions de l'archive
    metrics_data = compute_metrics_all_solutions(archive_unique, donnees)
    
    # Visualisation des métriques de l'archive
    plot_archive_metrics_visualization(metrics_data, archive_unique)
    
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"\nTemps d'exécution total : {exec_time:.2f} secondes")

    # Affichage graphique
    plot_final_results(archive_unique, hv_history, donnees, valid_solutions)

if __name__ == "__main__":
    main()
