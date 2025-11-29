import numpy as np
from video_scheduler import Solution

# Paramètres GWO
A_MAX = 2.0
A_MIN = 0.0

def compute_a(iteration, max_iter, a_max=A_MAX, a_min=A_MIN):
    """
    Calcule le paramètre a pour l'itération donnée.
    """
    return a_max - (a_max - a_min) * (iteration / max_iter)


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
        new_sol = Solution(new_assignment, donnees)
        new_sol.evaluate()

        new_population.append(new_sol)

    return new_population


# -------------------Lyliane -------------------

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


