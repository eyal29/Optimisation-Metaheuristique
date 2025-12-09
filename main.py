
 
# Redirection de la sortie standard vers un flux avec encodage UTF-8
import io
import sys
import time

from affichage import compute_and_display_metrics, display_archive, display_fronts, display_leaders, print_solution_info
from algo import ParetoArchive, check_early_stopping, compute_a, evaluate_and_filter_solutions, generate_fronts, select_leaders
from greedy import generate_greedy_solution
from metrics import compute_metrics_all_solutions
from initialization import initialize_algorithm
from utils_to_algo import gwo_update_population
from visualization import plot_archive_metrics_visualization, plot_final_results


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def setup_and_initialize(config_path="config.yaml"):
    """
    Orchestre l'initialisation de l'algorithme : chargement des données, 
    création de l'archive, et configuration des paramètres.
    """
    donnees, config, valid_solutions = initialize_algorithm(config_path)

    MAX_ITER = config["gwo"]["max_iter"]
    ARCHIVE_MAX = config["gwo"]["max_archive_size"]
    
    archive = ParetoArchive(max_size=ARCHIVE_MAX)
    
    # Paramètres d'arrêt précoce/historique
    hv_history = None
    ref_point = None
    iterations_without_improvement = 0
    prev_hv = None

    return donnees, config, valid_solutions, archive, hv_history, ref_point, iterations_without_improvement, prev_hv


def finalize_and_report(archive, donnees, valid_solutions, hv_history, start_time):
    """
    Gère l'exécution des Greedys, l'affichage final, le calcul des métriques et les visualisations.
    """
    # Nettoyage de l'archive
    archive_unique = archive.get_solutions()
    
    # =========================================================================
    # 🎯 ÉTAPE DE COMPARAISON AVEC L'ALGORITHME GREEDY (CALCUL & AFFICHAGE CONSOLE)
    # =========================================================================
    print("\n" + "="*70)
    print("COMPARAISON GWO-NSGA-II vs. ALGORITHMES GLOUTONS")
    print("="*70)
    
    # Exécution des stratégies Gloutonnes
    print("\n[Glouton] Génération des 3 solutions mono-objectives...")
    
    sol_greedy_m = generate_greedy_solution(donnees, greedy_mode='makespan')
    sol_greedy_c = generate_greedy_solution(donnees, greedy_mode='cost')
    sol_greedy_e = generate_greedy_solution(donnees, greedy_mode='energy')

    # Stocker les solutions Greedy avec un nom descriptif
    greedy_solutions = {
        'Greedy-Makespan': sol_greedy_m,
        'Greedy-Cost': sol_greedy_c,
        'Greedy-Energy': sol_greedy_e
    }
    
    # Affichage des solutions Gloutonnes
    print("\n--- SOLUTIONS GLOUTONNES ---")
    for name, sol in greedy_solutions.items():
        print(f"[{name}] Makespan={sol.makespan:.4f}, Cost={sol.cost:.4f}, Energy={sol.energy:.4f}")

    # =========================================================================
    # 📈 ÉTAPE DE RAPPORT FINAL (AFFICHAGE ET VISUALISATION)
    # =========================================================================
    
    # Affichage détaillé et résumé
    print("\n\n===== DÉTAILS DES SOLUTIONS FINALES DE L'ARCHIVE =====")
    for idx, solution in enumerate(archive_unique, 1):
        print_solution_info(solution, idx)
    
    display_archive(archive_unique, show_summary=True, valid_solutions=valid_solutions, donnees=donnees)
    
    # Calcul des métriques pour TOUTES les solutions de l'archive et visualisation
    metrics_data = compute_metrics_all_solutions(archive_unique, donnees)
    plot_archive_metrics_visualization(metrics_data, archive_unique)

    # Affichage du temps d'exécution
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"\nTemps d'exécution total : {exec_time:.2f} secondes")

    # Affichage graphique (Visualisation)
    plot_final_results(archive_unique, hv_history, donnees, valid_solutions, greedy_points=greedy_solutions)

    
def main():
    start_time = time.time()
    
    # 1. INITIALISATION ET SETUP
    donnees, config, valid_solutions, archive, hv_history, ref_point, iterations_without_improvement, prev_hv = \
        setup_and_initialize()

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
        evaluated_solutions = evaluate_and_filter_solutions(valid_solutions, donnees, POP_SIZE)
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
    finalize_and_report(archive, donnees, valid_solutions, hv_history, start_time)


if __name__ == "__main__":
    main()