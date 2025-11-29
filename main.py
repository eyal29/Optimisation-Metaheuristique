import time
from utils import load_data
from video_scheduler import Donnees, generate_valid_solution
from leader_pareto import generate_fronts, select_leaders, plot_fronts_3d
from gwo_utils import compute_a, gwo_update_population
from metrics import (
    extract_objectives,
    init_hv_tracking,
    update_hv_tracking,
    load_balancing_index,
    fog_utilization_ratio,
    energy_efficiency,
    average_latency,
    diversity_spread,
    pareto_size,
    spacing_metric,
    plot_hv_convergence,
    plot_metrics_bar,
    plot_pareto_2d,
)

MAX_ITER = 3 #10

if __name__ == "__main__":
    # Archive globale (front Pareto sur toutes les itérations)
    archive = []

    # === NOUVEAU lyliane === suivi HV
    hv_history = None
    ref_point = None

    videos, vms = load_data("datasets/videos.csv",
                            "datasets/machines_virtuelles.csv")
    donnees = Donnees(videos, vms)

    # Générer 10 solutions valides (on changera par 100 plus tard)
    valid_solutions = [generate_valid_solution(donnees) for _ in range(10)]#100

    # Archive globale (front Pareto sur toutes les itérations)
    archive = []
    start_time = time.time()

    #  BOUCLE PRINCIPALE 
    for t in range(MAX_ITER):
        print(f"\n===== ITERATION {t} =====")
        a = compute_a(t, MAX_ITER)  # on est obligé d'avoir a 
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

        #  Fronts de la pop actuelle 
        fronts = generate_fronts(valid_solutions)
        print("\n===== FRONTS NON DOMINES (population actuelle) =====")
        for i, front in enumerate(fronts):
            print(f"\nFront {i+1} :")
            for solution in front:
                idx = valid_solutions.index(solution) + 1
                print(f"  Solution {idx} :  Makespan={solution.makespan:.4f}, "
                      f"Cost={solution.cost:.4f}, Energy={solution.energy:.4f}")
                
        # MAJ de l'archive : on fusionne l'ancienne archive et la population actuelle
        all_solutions = archive + valid_solutions
        new_fronts = generate_fronts(all_solutions)  # On recalcule les fronts sur "tout ce qu'on connaît"
        archive = new_fronts[0]  # Le front 1 (non dominé) devient la nouvelle archive

        print("\n===== ARCHIVE (FRONT GLOBAL NON DOMINE) =====")
        for solution in archive:
            print(f"  Makespan={solution.makespan:.4f}, "
                  f"Cost={solution.cost:.4f}, Energy={solution.energy:.4f}")

        # === NOUVEAU lyliane : HV + diversité/spacing sur l'archive ===
        if len(archive) > 0:
            objs_arch = extract_objectives(archive)
            if hv_history is None:
                hv_history, ref_point = init_hv_tracking(archive)
            else:
                hv_history = update_hv_tracking(archive, hv_history, ref_point)

            div = diversity_spread(objs_arch)
            sp = spacing_metric(objs_arch)
            psize = pareto_size(objs_arch)
            print(f"\n[Metrics ARCHIVE] HV={hv_history[-1]:.4f}, "
                  f"Diversity={div:.4f}, Spacing={sp:.44f}, "
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

        # === NOUVEAU lyliane : métriques Fog/Cloud sur le leader alpha ===
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
    unique = {}
    for sol in archive:
        key = (sol.makespan, sol.cost, sol.energy)
        unique[key] = sol
    archive_unique = list(unique.values())

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

    # === NOUVEAU lyliane : plots de métriques ===
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