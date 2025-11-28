import numpy as np
from utils import load_data
from video_scheduler import Donnees, generate_valid_solution
from leader_pareto import generate_fronts, select_leaders, plot_fronts_3d
from gwo_utils import compute_a, gwo_update_population

MAX_ITER = 10

if __name__ == "__main__":
    videos, vms = load_data("datasets/videos.csv",
                            "datasets/machines_virtuelles.csv")
    donnees = Donnees(videos, vms)

    # Générer 10 solutions valides (on changera par 100 plus tard)
    valid_solutions = [generate_valid_solution(donnees) for _ in range(100)]

    # Archive globale (front Pareto sur toutes les itérations)
    archive = []
    
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

    # Front Pareto global basé sur l’archive
    archive_fronts = generate_fronts(archive_unique)

    # Affichage graphique des meilleures solutions (archive globale)
    plot_fronts_3d(archive_unique, archive_fronts)
