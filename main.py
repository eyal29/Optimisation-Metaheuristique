import time
from utils import load_config, load_data
from solutions_definiton import Donnees
from pareto import generate_fronts, select_leaders, ParetoArchive
from algo_utils import generate_valid_solution, compute_a
from utils_main import (
    evaluate_and_filter_solutions,
    display_fronts,
    display_archive,
    compute_and_display_archive_metrics,
    display_leaders,
    compute_and_display_alpha_metrics,
    update_and_filter_population,
    display_final_summary,
    plot_final_results,
    check_early_stopping
)

from metrics import (
    compute_metrics_all_solutions,
    plot_archive_metrics_visualization
)

import sys
import io
 
# Redirection de la sortie standard vers un flux avec encodage UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    config = load_config("config.yaml")

    MAX_ITER = config["gwo"]["max_iter"]
    POP_SIZE = config["gwo"]["population_size"]
    ARCHIVE_MAX = config["gwo"]["max_archive_size"]
    A_MAX = config["gwo"]["a_max"]
    A_MIN = config["gwo"]["a_min"]
    EARLY_STOPPING_THRESHOLD = config["gwo"]["early_stopping_threshold"]
    EARLY_STOPPING_PATIENCE = config["gwo"]["early_stopping_patience"]

    archive = ParetoArchive(max_size=ARCHIVE_MAX)
    hv_history = None
    ref_point = None

    videos_path = config["paths"]["videos"]
    vms_path = config["paths"]["vms"]

    videos, vms = load_data(videos_path, vms_path)
    donnees = Donnees(videos, vms)

    # Générer des solutions valides
    valid_solutions = [generate_valid_solution(donnees) for _ in range(POP_SIZE)]
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

        # Évaluation + filtrage par contrainte Lmax
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
        hv_history, ref_point = compute_and_display_archive_metrics(
            archive_solutions, hv_history, ref_point
        )

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
        compute_and_display_alpha_metrics(alpha, donnees)

        # Mise à jour de la population (GWO)
        valid_solutions = update_and_filter_population(
            evaluated_solutions, alpha, beta, delta, donnees, a, POP_SIZE
        )

    # Nettoyage de l'archive
    archive_unique = archive.get_solutions()

    # Résumé final
    display_final_summary(valid_solutions, archive_unique, donnees)
    
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
