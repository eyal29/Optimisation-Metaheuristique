import numpy as np
from utils import load_data
from video_scheduler import Donnees, generate_solutions, generate_valid_solution

# Création de l'instance des données
videos, vms = load_data("datasets/videos_reduit.csv","datasets/machines_virtuelles_reduit.csv")
donnees = Donnees(videos, vms)

for _ in range(10):
    valid_solutions = generate_valid_solution(donnees)

for idx, solution in enumerate(valid_solutions):
    print(f"Solution {idx + 1}:")
    print(f"  Affectation: {solution.assignment}")
    print(f"  Makespan: {solution.makespan:.3f}")
    print(f"  Cout: {solution.cost:.3f}")
    print(f"  Energie: {solution.energy:.3f}")
    print("----------")


print(valid_solutions[0].makespan)
