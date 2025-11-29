import copy
import matplotlib.pyplot as plt
import numpy as np
from leader_pareto import generate_fronts2
from gwo_utils import assign_crowding_distance

class ParetoArchive:
    def __init__(self, max_size=None):
        """
        max_size : limite optionnelle du nombre de solutions dans l'archive
                   (pour éviter explosions mémoire).
        """
        self.archive = []
        self.max_size = max_size

    def _dominates(self, a, b):
        """Test si a domine b (minimisation)."""
        return (
            (a.makespan <= b.makespan) and
            (a.cost     <= b.cost)     and
            (a.energy   <= b.energy)   and
            (
                a.makespan < b.makespan or
                a.cost     < b.cost     or
                a.energy   < b.energy
            )
        )

    def add(self, sol):
        """
        Ajoute une solution dans l'archive en :
          - retirant les solutions dominées,
          - ne dupliquant pas les équivalentes,
          - maintenant uniquement les non dominées.
        """

        # 1. Vérifier si elle est dominée par quelqu’un dans l’archive
        for s in self.archive:
            if self._dominates(s, sol):
                # Elle n'est pas meilleure → inutile de l’ajouter
                return False

        # 2. Retirer les solutions dominées par la nouvelle
        new_archive = []
        for s in self.archive:
            if not self._dominates(sol, s):  
                new_archive.append(s)

        self.archive = new_archive

        # 3. Ajouter la solution (copie profonde pour éviter les effets de bord)
        self.archive.append(copy.deepcopy(sol))

        # 4. Si l'archive dépasse la taille max → garder les plus "diversifiées"
        if self.max_size and len(self.archive) > self.max_size:
            # 4.1 Calcul des fronts Pareto de l'archive (indices)
            fronts = generate_fronts2(self.archive)

            # 4.2 Calcul des crowding distances
            crowding = assign_crowding_distance(self.archive, fronts)

            # 4.3 On trie les solutions par :
            #     1. Rang Pareto (front 1 avant front 2)
            #     2. Crowding distance décroissante
            # (comme dans NSGA-II)
            def sort_key(idx):
                # trouver le rang (front index)
                for f_index, front in enumerate(fronts):
                    if idx in front:
                        rank = f_index
                        break
                return (rank, -crowding[idx])  # rank croissant, crowding décroissant

            sorted_idx = sorted(range(len(self.archive)), key=sort_key)

            # On garde uniquement les max_size premiers indices
            selected_idx = sorted_idx[:self.max_size]

            # On reconstruit l’archive avec ces solutions
            self.archive = [self.archive[i] for i in selected_idx]

        return True

    def get_solutions(self):
        """Retourne une copie sécurisée."""
        return copy.deepcopy(self.archive)



