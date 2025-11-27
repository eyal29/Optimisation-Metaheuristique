import numpy as np
from utils import load_data
from video_scheduler import Donnees, generate_valid_solution
from leader_pareto import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

if __name__ == "__main__":

    videos, vms = load_data("datasets/videos_reduit.csv","datasets/machines_virtuelles_reduit.csv")
    donnees = Donnees(videos, vms)

    # Générer 10 solutions valides (on changera par 100 plus tard)
    valid_solutions = [generate_valid_solution(donnees) for _ in range(10)]

    for idx, solution in enumerate(valid_solutions):
        solution.evaluate()
        print(f"Solution {idx + 1}:")
        print(f"  Affectation: {solution.assignment}")
        print(f"  Makespan: {solution.makespan}")
        print(f"  Cout total: {solution.cost}")
        print(f"  Energie totale: {solution.energy}")
        print("----------")

    
    # ---- Affichage des fronts ----
    fronts = generate_fronts(valid_solutions)
    print("\n===== FRONTS NON DOMINES =====")
    for i, front in enumerate(fronts):
        print(f"\nFront {i+1} :")
        for solution in front:
            index = valid_solutions.index(solution) + 1 
            print(f"  Solution {index} :  Makespan={solution.makespan:.4f}, Cost={solution.cost:.4f}, Energy={solution.energy:.4f}")
    
    # ---- Affichage des leaders ----
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

    
    
        
# Affichage graphique 3D
plot_fronts_3d(valid_solutions, fronts)

   

