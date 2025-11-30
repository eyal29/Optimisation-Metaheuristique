import time
from utils import load_config, load_data
from solutions_definiton import Donnees
from pareto import generate_fronts, select_leaders, plot_fronts_3d, ParetoArchive
from algo_utils import gwo_update_population, generate_valid_solution, compute_a
from metrics import *

if __name__ == "__main__":
    config = load_config("config.yaml")

    MAX_ITER = config["gwo"]["max_iter"]
    POP_SIZE = config["gwo"]["population_size"]
    ARCHIVE_MAX = config["gwo"]["max_archive_size"]
    A_MAX = config["gwo"]["a_max"]
    A_MIN = config["gwo"]["a_min"]

    archive = ParetoArchive(max_size=ARCHIVE_MAX) # ou un autre nombre
    hv_history = None
    ref_point = None

    videos_path = config["paths"]["videos"]
    vms_path = config["paths"]["vms"]

    videos, vms = load_data(videos_path, vms_path)

    # on passera la vitesse de propagation au prochain point
    donnees = Donnees(videos, vms)

    # Générer des solutions valides
    valid_solutions = [generate_valid_solution(donnees) for _ in range(POP_SIZE)]#10 solutions au départ
    start_time = time.time()

    #  BOUCLE PRINCIPALE 
    for t in range(1, MAX_ITER + 1):
        print(f"\n===== ITERATION {t} =====")
        a = compute_a(t, MAX_ITER, a_max=A_MAX, a_min=A_MIN) 
        print(f"a = {a:.4f}")

        # Évaluation + affichage de la population
        for idx, solution in enumerate(valid_solutions):
            solution.evaluate()
            print(f"\nSolution {idx + 1}:")
            print(f"  Affectation: {solution.assignment}")
            print(f"  Makespan: {solution.makespan}")
            print(f"  Cout total: {solution.cost}")
            print(f"  Energie totale: {solution.energy}")
            print("----------")

        # Fronts de la pop actuelle 
        fronts = generate_fronts(valid_solutions)
        print("\n===== FRONTS NON DOMINES (population actuelle) =====")
        for i, front in enumerate(fronts):
            print(f"\nFront {i+1} :")
            for solution in front:
                idx = valid_solutions.index(solution) + 1
                print(f"  Solution {idx} :  Makespan={solution.makespan:.4f}, "
                      f"Cost={solution.cost:.4f}, Energy={solution.energy:.4f}")
                
        for sol in valid_solutions:
            archive.add(sol)

        print("\n===== ARCHIVE (FRONT GLOBAL NON DOMINE) =====")
        archive_solutions = archive.get_solutions()
        for sol in archive_solutions:
            print(f"  Makespan={sol.makespan:.4f}, "
                f"Cost={sol.cost:.4f}, Energy={sol.energy:.4f}")

        # HV + diversité/spacing sur l'archive ===
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

        #  Leaders de la population actuelle 
        alpha, beta, delta = select_leaders(valid_solutions)
        print("\n===== LEADERS =====")
        for name, solution in [("Alpha", alpha), ("Beta", beta), ("Delta", delta)]:
            if solution is None:
                print(f"{name} : None")
            else:
                idx = valid_solutions.index(solution) + 1
                print(f"{name} = Solution {idx} : "
                      f"Makespan={solution.makespan:.4f}, "
                      f"Cost={solution.cost:.4f}, "
                      f"Energy={solution.energy:.4f}")

        #  métriques Fog/Cloud sur le leader alpha ===
        if alpha is not None:
            lbi = load_balancing_index(alpha, donnees)
            fur = fog_utilization_ratio(alpha, donnees)
            ee = energy_efficiency(alpha)
            avg_lat = average_latency(alpha, donnees)
            print(f"\n[Metrics ALPHA] LBI={lbi:.4f}, FUR={fur:.4f}, "
                  f"EE={ee:.4f}, AvgLatency={avg_lat:.4f}")

        #  Mise à jour de la population (GWO)
        valid_solutions = gwo_update_population(
            valid_solutions, alpha, beta, delta, donnees, a
        )

    # Nettoyage de l’archive : suppression des doublons exacts
    archive_unique = archive.get_solutions()

    print("\n===== MEILLEURES SOLUTIONS (ARCHIVE FINALE SANS DOUBLONS) =====")
    for sol in archive_unique:
        print(f"  Makespan={sol.makespan:.4f}, "
              f"Cost={sol.cost:.4f}, Energy={sol.energy:.4f}")
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"\nTemps d'exécution total : {exec_time:.2f} secondes")

    # Front Pareto global basé sur l’archive
    archive_fronts = generate_fronts(archive_unique)

    # Affichage graphique des meilleures solutions (archive globale)
    plot_fronts_3d(archive_unique, archive_fronts)

    # 1) Convergence HV
    if hv_history is not None:
        plot_hv_convergence(hv_history, title="Convergence de l'hypervolume (GWO Fog-Cloud)")

    # 2) Barplot des métriques Fog/Cloud sur une solution "représentative"
    if len(archive_unique) > 0:
        # exemple : solution avec makespan minimal dans l'archive
        best = min(archive_unique, key=lambda s: s.makespan)

        metrics_dict = {
            "LBI": load_balancing_index(best, donnees),
            "FUR": fog_utilization_ratio(best, donnees),
            "EE": energy_efficiency(best),
            "Avg Latency": average_latency(best, donnees),
        }
        plot_metrics_bar(metrics_dict, title="Métriques Fog-Cloud (meilleure solution archive)")

        # 3) Pareto 2D Makespan vs Cost (couleur = Energy)
        objs_arch_final = extract_objectives(archive_unique)
        plot_pareto_2d(
            objs_arch_final,
            x_idx=0, y_idx=1,
            x_label="Makespan", y_label="Cost"
        )