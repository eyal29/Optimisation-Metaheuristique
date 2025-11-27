import numpy as np
from utils import load_data
from video_scheduler import Donnees, generate_valid_solution

if __name__ == "__main__":

    # Création de l'instance des données
    videos, vms = load_data("datasets/videos.csv","datasets/machines_virtuelles.csv")
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

