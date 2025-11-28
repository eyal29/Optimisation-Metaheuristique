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
