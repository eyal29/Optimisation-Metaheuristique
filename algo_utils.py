import numpy as np
from solutions_definiton import Solution

# Paramètres GWO
A_MAX = 2.0
A_MIN = 0.0
def compute_a(iteration, max_iter, a_max=A_MAX, a_min=A_MIN):
    """
    Calcule le paramètre a pour l'itération donnée.
    """
    return a_max - (a_max - a_min) * (iteration / max_iter)


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


def gwo_update_population(population, alpha, beta, delta, donnees, a):
    """Met à jour la population selon les leaders GWO."""
    
    new_population = []
    n = donnees.n      # nombre de vidéos
    p = donnees.p      # nombre de VMs

    for solution in population:
        if solution in (alpha, beta, delta):  # Garder les leaders tels quels
            new_population.append(solution)
            continue

        # Nouvelle affectation continue
        new_assignment = solution.assignment.astype(float).copy()

        for i in range(n):
            Xi = solution.assignment[i]
            X_alpha = alpha.assignment[i]
            X_beta  = beta.assignment[i]
            X_delta = delta.assignment[i]

            #  Alpha 
            r1, r2 = np.random.rand(), np.random.rand()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * X_alpha - Xi)
            X1 = X_alpha - A1 * D_alpha

            #  Beta
            r1, r2 = np.random.rand(), np.random.rand()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * X_beta - Xi)
            X2 = X_beta - A2 * D_beta

            # Delta 
            r1, r2 = np.random.rand(), np.random.rand()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * X_delta - Xi)
            X3 = X_delta - A3 * D_delta

            # Moyenne des leaders 
            X_new = (X1 + X2 + X3) / 3.0

            #  Passage aux new solutions avec les valeurs 
            vm_new = int(round(X_new))
            vm_new = max(0, min(p - 1, vm_new))

            new_assignment[i] = vm_new

                # Création nouvelle solution
        new_assignment = new_assignment.astype(int)

        # 🔧 Réparation mémoire : s'assurer qu'on ne dépasse pas les capacités des VMs
        new_assignment = repair_assignment(new_assignment, donnees)

        new_sol = Solution(new_assignment, donnees)
        new_sol.evaluate()

        new_population.append(new_sol)

    return new_population


# ------------------- Réparation mémoire après GWO -------------------

def repair_assignment(assignment, donnees):
    """
    Répare une affectation qui pourrait violer les capacités mémoire des VMs.

    - assignment : vecteur des VMs pour chaque vidéo (taille n)
    - donnees    : objet Donnees (contient m_i, memory_capacity, U_ij, n, p)

    Principe :
      1) on calcule la mémoire utilisée par VM
      2) pour chaque VM qui dépasse sa capacité :
         - on déplace certaines vidéos vers des VMs qui ont encore de la place
         - on choisit la VM cible qui donne le plus petit U_ij (temps le plus faible)
    """
    assignment = assignment.copy().astype(int)
    n, p = donnees.n, donnees.p
    m_i = donnees.m_i
    capacity = donnees.memory_capacity
    U_ij = donnees.U_ij

    # 1. Calcul de la mémoire utilisée par VM
    memory_used = np.zeros(p, dtype=float)
    for i in range(n):
        vm = assignment[i]
        memory_used[vm] += m_i[i]

    # 2. Pour chaque VM qui dépasse sa capacité → on tente de déplacer des vidéos
    for vm in range(p):
        # Tant que cette VM est surchargée
        while memory_used[vm] > capacity[vm]:
            # vidéos actuellement sur cette VM
            videos_on_vm = [i for i in range(n) if assignment[i] == vm]
            if not videos_on_vm:
                break  # plus rien à déplacer

            # On essaye de déplacer d'abord les vidéos les plus "lourdes" en mémoire
            videos_on_vm.sort(key=lambda i: m_i[i], reverse=True)

            moved = False
            for i in videos_on_vm:
                # chercher des VMs cibles possibles
                candidates = [
                    k for k in range(p)
                    if k != vm and memory_used[k] + m_i[i] <= capacity[k]
                ]
                if not candidates:
                    continue

                # parmi les VMs possibles, on prend celle avec le plus petit temps U_ij
                best_vm = min(candidates, key=lambda k: U_ij[i, k])

                # déplacer la vidéo i de vm → best_vm
                assignment[i] = best_vm
                memory_used[vm]      -= m_i[i]
                memory_used[best_vm] += m_i[i]
                moved = True
                break  # on re-vérifie la surcharge de la VM

            if not moved:
                # Impossible de corriger plus cette VM (pas de place ailleurs)
                # On sort de la boucle pour éviter une boucle infinie
                break

    return assignment


def crowding_distance(front_objs: np.ndarray) -> np.ndarray:
    """
    Compute crowding distance for a single Pareto front.
    
    Parameters
    ----------
    front_objs : np.ndarray
        Objective values for solutions in one front, shape (N, M)
    
    Returns
    -------
    distances : np.ndarray of shape (N,)
    """
    N, M = front_objs.shape
    distances = np.zeros(N, dtype=float)

    if N == 1:
        distances[0] = np.inf
        return distances

    if N == 2:
        distances[:] = np.inf
        return distances

    for m in range(M):
        sorted_idx = np.argsort(front_objs[:, m])
        sorted_vals = front_objs[sorted_idx, m]

        fmin = sorted_vals[0]
        fmax = sorted_vals[-1]

        if fmax - fmin == 0:
            continue

        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf

        for i in range(1, N - 1):
            distances[sorted_idx[i]] += (sorted_vals[i+1] - sorted_vals[i-1]) / (fmax - fmin)

    return distances



def assign_crowding_distance(population, pareto_fronts):
    """
    Calcule la crowding-distance pour toute la population,
    en appliquant crowding_distance() front par front.

    population : liste d'objets Solution
    pareto_fronts : liste de listes d’indices (F1, F2, F3…)
    """

    N = len(population)
    crowding = np.zeros(N)

    # On extrait les objectifs dans une matrice (N, M)
    objs = np.array([
        [sol.makespan, sol.cost, sol.energy]
        for sol in population
    ])

    for front in pareto_fronts:
        if len(front) == 0:
            continue

        # extraire les valeurs objectives du front
        front_objs = objs[front]

        # calculer la distance pour ce front
        cd = crowding_distance(front_objs)

        # remettre les valeurs au bon endroit
        for i, idx in enumerate(front):
            crowding[idx] = cd[i]

    return crowding


def check_lmax_constraint(solution, donnees):
    """
    Vérifie si une solution respecte la contrainte de latence maximale Lmax.
    
    Pour chaque VM j : sum_i(U_ij * x_ij) <= L_max
    où U_ij est la latence totale pour la vidéo i sur VM j.
    
    Retourne : (bool, dict)
        - bool : True si valide, False sinon
        - dict : infos sur violations (seuil, latences par VM, violations)
    """
    try:
        from utils import load_config
        config = load_config("config.yaml")
        lmax_config = config.get("constraints", {}).get("lmax", False)
    except Exception:
        return True, None
    
    # Si lmax est False, pas de contrainte
    if lmax_config is False or lmax_config == "false":
        return True, None
    
    # Si lmax est True, valeur par défaut
    if lmax_config is True or lmax_config == "true":
        lmax_threshold = 100.0
    else:
        try:
            lmax_threshold = float(lmax_config)
        except (ValueError, TypeError):
            return True, None
    
    # Calculer latences par VM
    U = donnees.U_ij
    n = donnees.n
    p = donnees.p
    
    vm_latencies = np.zeros(p)
    for i in range(n):
        vm_idx = int(solution.assignment[i])
        vm_latencies[vm_idx] += U[i, vm_idx]
    
    # Déterminer violations
    violations = []
    for j in range(p):
        if vm_latencies[j] > lmax_threshold:
            violations.append((j, vm_latencies[j]))
    
    info = {
        "threshold": lmax_threshold,
        "vm_latencies": vm_latencies.tolist(),
        "violations": violations,
        "is_valid": len(violations) == 0
    }
    
    return len(violations) == 0, info


