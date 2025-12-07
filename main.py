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
    plot_final_results
)


def main():
    config = load_config("config.yaml")

    MAX_ITER = config["gwo"]["max_iter"]
    POP_SIZE = config["gwo"]["population_size"]
    ARCHIVE_MAX = config["gwo"]["max_archive_size"]
    A_MAX = config["gwo"]["a_max"]
    A_MIN = config["gwo"]["a_min"]

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

        # Métriques sur l'archive
        hv_history, ref_point = compute_and_display_archive_metrics(archive_solutions, hv_history, ref_point)

        # Sélection des leaders
        alpha, beta, delta = select_leaders(evaluated_solutions)
        display_leaders(evaluated_solutions, alpha, beta, delta)

        # Métriques sur le leader alpha
        compute_and_display_alpha_metrics(alpha, donnees)

        # Mise à jour de la population (GWO)
        valid_solutions = update_and_filter_population(evaluated_solutions, alpha, beta, delta, donnees, a, POP_SIZE)

    # Nettoyage de l'archive
    archive_unique = archive.get_solutions()

    # Résumé final
    display_final_summary(valid_solutions, archive_unique, donnees)
    
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"\nTemps d'exécution total : {exec_time:.2f} secondes")

    # Affichage graphique
    plot_final_results(archive_unique, hv_history, donnees, valid_solutions)


if __name__ == "__main__":
    main()
