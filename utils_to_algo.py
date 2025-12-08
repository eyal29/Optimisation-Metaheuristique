import numpy as np
import pandas as pd

class Donnees:
    #pour recuperer les donnees des datasets et calculer la matrice Uij
    def __init__(self, videos: pd.DataFrame, vms: pd.DataFrame):
        # Vidéos
        self.n = len(videos)
        self.m_i = videos["Memory required (MB)"].to_numpy(dtype=float)
        self.q_i = videos["Input file size (MB)"].to_numpy(dtype=float)
        self.w_i = videos["wi"].to_numpy(dtype=float)
        self.q_out = videos["Output file size (MB)"].to_numpy(dtype=float)
        

        if "type" not in vms.columns:
            raise ValueError(
                "La colonne 'type' est absente dans machines_virtuelles.csv. "
                "Elle est nécessaire pour calculer FUR."
            )

        # Nettoyage du format (au cas où)
        type_col = vms["type"].astype(str).str.strip().str.lower()

        # Bool : True si Fog, False sinon
        self.is_fog_j = type_col.eq("fog").values

        # VMs
        self.p = len(vms)
        self.P_j = vms["cpu_power_MIPS"].to_numpy(dtype=float)
        self.lambda_j = vms["lambda (s)"].to_numpy(dtype=float)
        self.beta_j = vms["beta (MB)"].to_numpy(dtype=float)
        self.gamma_j = vms["gamma (MB)"].to_numpy(dtype=float)
        self.energy_j = vms["P_energy (Watts)"].to_numpy(dtype=float)
        self.Dij = vms["Dij (MBps)"].to_numpy(dtype=float)
        self.distances = vms["distance (km)"].to_numpy(dtype=float)
        self.memory_capacity = vms["memory_capacity (MB)"].to_numpy(dtype=float)
        self.U_ij = self.compute_Uij()

    def compute_Uij(self) -> np.ndarray:
   
        U_ij = np.zeros((self.n, self.p), dtype=float)
        c = 300000 

        for i in range(self.n):
            for j in range(self.p):
                L_transfert = (2 * self.distances[j]) / c + (self.q_i[i] / self.Dij[j])
                processing_time = self.w_i[i] / self.P_j[j]
                L_sortie = self.q_out[i] / self.Dij[j]

                U_ij[i, j] = L_transfert + processing_time + L_sortie

        return U_ij

class Solution:
    def __init__(self, assignment, donnees: Donnees):
    
        self.assignment = assignment  # Affectation des vidéos aux VMs
        self.donnees = donnees        # L'instance du problème
        self.makespan = None
        self.cost = None
        self.energy = None

    def evaluate(self):
        n, p = self.donnees.n, self.donnees.p  # n = nombre de vidéos, p = nombre de VMs
        U_ij = self.donnees.U_ij  # Matrice des temps d'exécution entre vidéos et VMs

        # 1. Calcul du makespan (temps de traitement total)
        load = np.zeros(p, dtype=float)  # Charge par machine virtuelle
        for i in range(n):
            vm = self.assignment[i]  # VM à laquelle la vidéo i est assignée
            load[vm] += U_ij[i, vm]  # Charge totale pour cette VM
        # Le makespan est la machine ayant la charge maximale
        self.makespan = load.max()

        # 2. Calcul du coût
        total_cost = 0.0
        for i in range(n):
            j = self.assignment[i]  # VM assignée à la vidéo i
            U = U_ij[i, j]
            m = self.donnees.m_i[i]  # Mémoire utilisée par la vidéo i
            q = self.donnees.q_i[i]  # Taille des données entrantes de la vidéo i
            lambda_j = self.donnees.lambda_j[j]
            beta_j = self.donnees.beta_j[j]
            gamma_j = self.donnees.gamma_j[j]
            total_cost += (lambda_j * U + beta_j * m + gamma_j * q)
        self.cost = total_cost
        
        # 3. Calcul de l'énergie
        total_energy = 0
        for j in range(p):
            total_energy += load[j] * self.donnees.energy_j[j]  # Charge par VM * énergie de la VM

        self.energy = total_energy

# FONCTIONS DE DOMINANCE ET SIGNATURE
def sol_signature(sol):
    return (round(sol.makespan, 4), round(sol.cost, 4), round(sol.energy, 4))

def dominates(a, b):
    return (
        a.makespan <= b.makespan and
        a.cost <= b.cost and
        a.energy <= b.energy and
        (a.makespan < b.makespan or a.cost < b.cost or a.energy < b.energy)
    )

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
        from initialization import load_config
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
