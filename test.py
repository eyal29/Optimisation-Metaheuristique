import numpy as np
from utils import load_data
from video_scheduler import Donnees, Solution

# Création de l'instance du problème
videos, vms = load_data("datasets/videos.csv","datasets/machines_virtuelles.csv")
donnees = Donnees(videos, vms)

# Création d'une solution avec une affectation aléatoire
assignment = np.random.randint(0, donnees.p, size=donnees.n)  # Affectation aléatoire
solution = Solution(assignment, donnees)
solution.evaluate()

print("Makespan:", solution.makespan)
print("Coût:", solution.cost)
print("Énergie:", solution.energy)