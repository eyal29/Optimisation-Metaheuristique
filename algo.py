import copy
import numpy as np
from utils_to_algo import assign_crowding_distance, crowding_distance, Solution, dominates, sol_signature


def generate_valid_solution(donnees):
    assignment = np.zeros(donnees.n, dtype=int)  
    memory_used = np.zeros(donnees.p, dtype=float) 
    
    for i in range(donnees.n):
        valid_assignment = False
        while not valid_assignment:
            vm = np.random.randint(0, donnees.p)  # Choisir une VM aléatoire
            if memory_used[vm] + donnees.m_i[i] <= donnees.memory_capacity[vm]:
                # Affecter la vidéo à cette VM
                assignment[i] = vm
                memory_used[vm] += donnees.m_i[i]  
                valid_assignment = True  
    
    solution = Solution(assignment, donnees)  # Créer un objet Solution avec l'affectation
    return solution

def compute_a(iteration, max_iter, a_max, a_min):
    return a_max - (a_max - a_min) * (iteration / max_iter)


def evaluate_solutions(valid_solutions, donnees, POP_SIZE):
    evaluated_solutions = []

    for solution in valid_solutions:
        solution.evaluate()
        evaluated_solutions.append(solution)

    print(f"\nSolutions évaluées: {len(evaluated_solutions)}/{POP_SIZE}")
    
    return evaluated_solutions


# GÉNÉRATION DES FRONTS DE PARETO
def generate_fronts(population, return_indices=False):
    remaining = list(range(len(population)))
    fronts_idx = []

    while remaining:
        current_front = []
        for i in remaining:
            if not any(dominates(population[j], population[i]) 
                      for j in remaining if i != j):
                current_front.append(i)

        fronts_idx.append(current_front)
        remaining = [i for i in remaining if i not in current_front]

    if return_indices:
        return fronts_idx

    return [[population[i] for i in front] for front in fronts_idx]



# ARCHIVE DE PARETO
class ParetoArchive:
    def __init__(self, max_size=None):
        self.archive = []
        self.max_size = max_size

    def _dominates(self, a, b):
        return (
            a.makespan <= b.makespan and
            a.cost <= b.cost and
            a.energy <= b.energy and
            (a.makespan < b.makespan or a.cost < b.cost or a.energy < b.energy)
        )

    def add(self, sol):
 
        # 1) Si sol est dominée par une solution de l’archive → on la rejette
        for s in self.archive:
            if self._dominates(s, sol):
                return False
 
        # 2) Retirer les solutions dominées par sol
        new_archive = []
        for s in self.archive:
            if not self._dominates(sol, s):
                new_archive.append(s)
 
        # 3) Éviter les doublons (mêmes objectifs / même signature)
        sig_new = sol_signature(sol)
        for s in new_archive:
            if sol_signature(s) == sig_new:
                self.archive = new_archive
                return False
 
        # 4) Ajouter la nouvelle solution (non dominée, non dupliquée)
        new_archive.append(copy.deepcopy(sol))
        self.archive = new_archive
 
        # 5) Si pas de limite de taille → on s'arrête là
        if self.max_size is None or len(self.archive) <= self.max_size:
            return True
 
        # 6) Tri NSGA-II (rang Pareto + crowding distance)
        fronts = generate_fronts(self.archive, return_indices=True)
        first_front = fronts[0]  
        crowding = assign_crowding_distance(self.archive, [first_front])
 
        if self.max_size is None or len(first_front) <= self.max_size:
            selected_idx = first_front
        else:

            selected_idx = sorted(
                first_front,
                key=lambda idx: -crowding[idx]   # crowding décroissant
            )[:self.max_size]
 
        # On reconstruit l’archive avec uniquement les solutions du front 1 sélectionné
        self.archive = [self.archive[i] for i in selected_idx]
        return True

    def get_solutions(self):
        return copy.deepcopy(self.archive)
    

def check_early_stopping(hv_history, prev_hv, iterations_without_improvement, 
                        early_stopping_threshold, early_stopping_patience):
    """    
    Args:
        hv_history: Historique des hypervolumes
        prev_hv: Hypervolume précédent
        iterations_without_improvement: Nombre d'itérations sans amélioration
        early_stopping_threshold: Seuil d'amélioration minimale
        early_stopping_patience: Nombre d'itérations avant arrêt
    """
    if not hv_history:
        return False, iterations_without_improvement, prev_hv
    
    current_hv = hv_history[-1]
    
    if prev_hv is not None:
        improvement = (current_hv - prev_hv) / prev_hv
        
        if improvement < early_stopping_threshold:
            iterations_without_improvement += 1
            print(f"⚠️  HV n'a pas amélioré de {early_stopping_threshold*100}% "
                  f"(amélioration: {improvement*100:.4f}%)")
            print(f"   Itérations sans amélioration: "
                  f"{iterations_without_improvement}/{early_stopping_patience}")
            
            if iterations_without_improvement >= early_stopping_patience:
                print(f"\n ARRÊT PRÉCOCE: Pas d'amélioration depuis "
                      f"{early_stopping_patience} itérations")
                print(f"   HV final: {current_hv:.4f}")
                return True, iterations_without_improvement, current_hv
        else:
            iterations_without_improvement = 0
            print(f"Amélioration détectée: {improvement*100:.4f}%")
    
    return False, iterations_without_improvement, current_hv



# SÉLECTION DES LEADERS
def select_leaders(population):
    if not population:
        return None, None, None

    fronts = generate_fronts(population)
    if not fronts or not fronts[0]:
        return None, None, None

    F1 = fronts[0] 

    # Cas normal : au moins 3 solutions dans F1
    if len(F1) >= 3:
        front_objs = np.array([[s.makespan, s.cost, s.energy] for s in F1], dtype=float)
        cd = crowding_distance(front_objs)
        sorted_idx = np.argsort(-cd)
        
        return F1[sorted_idx[0]], F1[sorted_idx[1]], F1[sorted_idx[2]]

    # Cas rare : F1 contient moins de 3 solutions
    leaders = []
    score = lambda s: s.makespan + s.cost + s.energy
    
    for front in fronts:
        for s in sorted(front, key=score):
            leaders.append(s)
            if len(leaders) == 3:
                return leaders[0], leaders[1], leaders[2]

    # Si moins de 3 solutions au total
    while len(leaders) < 3:
        leaders.append(None)

    return leaders[0], leaders[1], leaders[2]



