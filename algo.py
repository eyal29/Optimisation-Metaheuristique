import copy
import numpy as np
from utils_to_algo import assign_crowding_distance, check_lmax_constraint, crowding_distance, Solution, dominates, gwo_update_population, sol_signature
# from utils_main import print_solution_info


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
    """
    Calcule le paramètre a pour l'itération donnée.
    """
    return a_max - (a_max - a_min) * (iteration / max_iter)


def evaluate_and_filter_solutions(valid_solutions, donnees, POP_SIZE):
    """
    Évalue les solutions et filtre celles qui respectent la contrainte Lmax.
    
    Args:
        valid_solutions: Liste des solutions à évaluer
        donnees: Données du problème
        POP_SIZE: Taille de la population
    
    Returns:
        Liste des solutions évaluées et valides
    """
    evaluated_solutions = []
    rejected_count = 0
    
    for idx, solution in enumerate(valid_solutions, 1):
        solution.evaluate()
        lmax_valid, lmax_info = check_lmax_constraint(solution, donnees)
        solution.lmax_valid = lmax_valid
        solution.lmax_info = lmax_info
        
        # print_solution_info(solution, idx, lmax_valid, lmax_info)
        
        if lmax_valid:
            evaluated_solutions.append(solution)
        else:
            print(" Solution rejetée (ne respecte pas Lmax)")
            rejected_count += 1
    
    # Si aucune solution valide, générer de nouvelles solutions
    if not evaluated_solutions:
        print("ATTENTION: Aucune solution ne respecte Lmax! Génération de nouvelles solutions.")
        evaluated_solutions = [generate_valid_solution(donnees) for _ in range(len(valid_solutions))]
        for sol in evaluated_solutions:
            sol.evaluate()
    
    print(f"\nSolutions acceptées: {len(evaluated_solutions)}/{POP_SIZE}, "
          f"rejetées: {POP_SIZE - len(evaluated_solutions)}")
    
    return evaluated_solutions


# GÉNÉRATION DES FRONTS DE PARETO
def generate_fronts(population, return_indices=False):
    """Calcule les fronts de Pareto par dominance successive.

    Args:
        population: Liste de solutions
        return_indices: Si True, retourne des listes d'indices au lieu de solutions

    Returns:
        Liste de fronts, chaque front étant une liste de solutions ou d'indices
    """
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
    """Archive maintenant uniquement les solutions non dominées."""

    def __init__(self, max_size=None):
        self.archive = []
        self.max_size = max_size

    def _dominates(self, a, b):
        """Teste si a domine b (minimisation sur tous les objectifs) """
        return (
            a.makespan <= b.makespan and
            a.cost <= b.cost and
            a.energy <= b.energy and
            (a.makespan < b.makespan or a.cost < b.cost or a.energy < b.energy)
        )

    def add(self, sol):
        """
        Ajoute une solution dans l'archive :
          - retire les solutions dominées,
          - pas de duplicata => uniquement les non dominés 
          - réordonnant ensuite l'archive selon NSGA-II
        """
        # print(" ➤ Tentative d'ajout d'une nouvelle solution :")
        # print(f"     Taille archive AVANT ajout : {len(self.archive)}")

        # # Si sol est dominée par une solution de l’archive → on la rejette
        # for s in self.archive:
        #     if self._dominates(s, sol):
        #         print("   ✘ SOLUTION REJETÉE : elle est dominée par une solution de l'archive.")
        #         print(f"     Taille archive APRÈS tentative : {len(self.archive)}")
        #         return False

        # Retirer les solutions dominées par sol
        new_archive = []
        removed = 0
        for s in self.archive:
            if self._dominates(sol, s):
                removed += 1
            else:
                new_archive.append(s)

        if removed > 0:
            print(f"   ✔ {removed} solution(s) dominée(s) supprimée(s) de l'archive.")
        else:
            print("   • Aucune solution supprimée (aucune dominée par la nouvelle).")

        #Éviter les doublons (mêmes objectifs)
        sig_new = sol_signature(sol)
        for s in new_archive:
            if sol_signature(s) == sig_new:
                print("   ✘ SOLUTION NON AJOUTÉE : déjà présente (doublon).")
                self.archive = new_archive
                return False

        # Ajouter la solution
        new_archive.append(copy.deepcopy(sol))
        self.archive = new_archive
        print("   ✔ SOLUTION AJOUTÉE à l'archive (avant tri NSGA-II).")
    
        # Fronts Pareto sur toute l'archive (indices)
        fronts = generate_fronts(self.archive, return_indices=True)

        # Crowding distance pour toutes les solutions
        crowding = assign_crowding_distance(self.archive, fronts)

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
        self.max_size = None


        # On reconstruit l’archive avec ces solutions
        self.archive = [self.archive[i] for i in selected_idx]

    def get_solutions(self):
        """Retourne une copie sécurisée de l'archive."""
        return copy.deepcopy(self.archive)
    

def check_early_stopping(hv_history, prev_hv, iterations_without_improvement, 
                        early_stopping_threshold, early_stopping_patience):
    """
    Vérifie si l'arrêt précoce doit être déclenché basé sur la stagnation de l'hypervolume.
    
    Args:
        hv_history: Historique des hypervolumes
        prev_hv: Hypervolume précédent
        iterations_without_improvement: Nombre d'itérations sans amélioration
        early_stopping_threshold: Seuil d'amélioration minimale
        early_stopping_patience: Nombre d'itérations avant arrêt
    
    Returns:
        Tuple (should_stop, iterations_count, prev_hv)
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

    F1 = fronts[0]  # Front 1 = solutions non dominées

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


def update_and_filter_population(evaluated_solutions, alpha, beta, delta, donnees, a, POP_SIZE):
    """
    Met à jour la population avec GWO et filtre les solutions valides.
    
    Args:
        evaluated_solutions: Population actuelle
        alpha, beta, delta: Leaders de la meute
        donnees: Données du problème
        a: Paramètre de convergence GWO
        POP_SIZE: Taille de la population
    
    Returns:
        Liste des solutions valides après mise à jour
    """
    new_population = gwo_update_population(evaluated_solutions, alpha, beta, delta, donnees, a)
    
    valid_solutions = [sol for sol in new_population 
                       if check_lmax_constraint(sol, donnees)[0]]
    
    # Générer de nouvelles solutions si nécessaire
    max_attempts = 1000
    attempts = 0
    
    while len(valid_solutions) < POP_SIZE and attempts < max_attempts:
        new_sol = generate_valid_solution(donnees)
        if check_lmax_constraint(new_sol, donnees)[0]:
            valid_solutions.append(new_sol)
        attempts += 1
    
    if len(valid_solutions) < POP_SIZE:
        print(f"\n⚠️  ATTENTION: Impossible de trouver {POP_SIZE} solutions valides!")
        print(f"   Seuil Lmax trop restrictif? Trouvé: {len(valid_solutions)}/{POP_SIZE}")
        print(f"   Utilisation de {len(valid_solutions)} solutions pour l'itération suivante.")
    
    return valid_solutions

