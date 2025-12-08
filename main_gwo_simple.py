"""
GWO mono-objectif pour comparaison avec MOGWO.
Optimise une fonction de coût composite : makespan + cost + energy
"""



# =============================================================================
# FONCTION OBJECTIF COMPOSITE
# =============================================================================

import time
from matplotlib import pyplot as plt
import numpy as np

from algo import compute_a, generate_valid_solution
from initialization import load_config, load_data
from utils_to_algo import Donnees, Solution, check_lmax_constraint, repair_assignment


def composite_objective(solution):
    """
    Fonction objectif composite à minimiser.
    Plus la valeur est petite, meilleure est la solution.
    """
    # Évaluer la solution si nécessaire
    if solution.makespan is None:
        solution.evaluate()
    
    # Normalisation simple : somme pondérée
    return solution.makespan + solution.cost + solution.energy


# =============================================================================
# SÉLECTION DES LEADERS GWO MONO-OBJECTIF
# =============================================================================

def select_leaders_mono(population):
    """
    Sélectionne alpha, beta, delta basés sur la fonction objectif composite.
    
    Returns:
        tuple: (alpha, beta, delta) - les 3 meilleures solutions
    """
    if len(population) < 3:
        # Cas limite
        sorted_pop = sorted(population, key=composite_objective)
        alpha = sorted_pop[0] if len(sorted_pop) > 0 else None
        beta = sorted_pop[1] if len(sorted_pop) > 1 else None
        delta = sorted_pop[2] if len(sorted_pop) > 2 else None
        return alpha, beta, delta
    
    # Tri par fonction objectif croissante
    sorted_pop = sorted(population, key=composite_objective)
    
    return sorted_pop[0], sorted_pop[1], sorted_pop[2]


# =============================================================================
# MISE À JOUR GWO MONO-OBJECTIF
# =============================================================================

def gwo_update_mono(population, alpha, beta, delta, donnees, a):
    """
    Mise à jour GWO classique sans considération Pareto.
    """
    new_population = []
    n = donnees.n
    p = donnees.p

    for solution in population:
        # Garder les leaders tels quels
        if solution in (alpha, beta, delta):
            new_population.append(solution)
            continue

        # Nouvelle affectation
        new_assignment = np.zeros(n, dtype=int)

        for i in range(n):
            Xi = solution.assignment[i]
            X_alpha = alpha.assignment[i]
            X_beta = beta.assignment[i]
            X_delta = delta.assignment[i]

            # Alpha
            r1, r2 = np.random.rand(), np.random.rand()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * X_alpha - Xi)
            X1 = X_alpha - A1 * D_alpha

            # Beta
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

            # Moyenne des 3 leaders
            X_new = (X1 + X2 + X3) / 3.0

            # Arrondir et borner
            vm_new = int(round(X_new))
            vm_new = max(0, min(p - 1, vm_new))
            new_assignment[i] = vm_new

        # Réparation mémoire
        new_assignment = repair_assignment(new_assignment, donnees)
        
        # Créer nouvelle solution (assignment, donnees)
        new_sol = Solution(new_assignment, donnees)
        new_population.append(new_sol)

    return new_population


# =============================================================================
# FILTRAGE PAR CONTRAINTE Lmax
# =============================================================================

def filter_by_lmax(population, donnees):
    """Filtre les solutions qui violent la contrainte Lmax."""
    valid = []
    for sol in population:
        lmax_valid, lmax_info = check_lmax_constraint(sol, donnees)
        if lmax_valid:
            valid.append(sol)
    return valid


# =============================================================================
# MAIN GWO MONO-OBJECTIF
# =============================================================================

def main():
    config = load_config("config.yaml")

    MAX_ITER = config["gwo"]["max_iter"]
    POP_SIZE = config["gwo"]["population_size"]
    A_MAX = config["gwo"]["a_max"]
    A_MIN = config["gwo"]["a_min"]

    videos_path = config["paths"]["videos"]
    vms_path = config["paths"]["vms"]

    videos, vms = load_data(videos_path, vms_path)
    donnees = Donnees(videos, vms)

    # Générer population initiale
    population = [generate_valid_solution(donnees) for _ in range(POP_SIZE)]
    start_time = time.time()

    # Historique de la meilleure solution
    best_history = []
    best_makespan_history = []
    best_cost_history = []
    best_energy_history = []

    print("\n" + "="*80)
    print("GWO MONO-OBJECTIF - Optimisation de la fonction composite")
    print("="*80)

    # BOUCLE PRINCIPALE
    for t in range(1, MAX_ITER + 1):
        print(f"\n===== ITERATION {t} =====")
        a = compute_a(t, MAX_ITER, a_max=A_MAX, a_min=A_MIN)
        print(f"a = {a:.4f}")

        # Filtrage par contrainte Lmax
        valid_population = filter_by_lmax(population, donnees)
        
        if not valid_population:
            print("⚠️ Aucune solution valide, régénération...")
            population = [generate_valid_solution(donnees) for _ in range(POP_SIZE)]
            continue

        # Compléter si nécessaire
        while len(valid_population) < POP_SIZE:
            valid_population.append(generate_valid_solution(donnees))

        # Sélection des leaders
        alpha, beta, delta = select_leaders_mono(valid_population)

        # Afficher les leaders
        alpha_obj = composite_objective(alpha)
        print(f"\n🐺 Alpha: Objectif={alpha_obj:.2f} "
              f"(M={alpha.makespan:.2f}, C={alpha.cost:.2f}, E={alpha.energy:.2f})")
        print(f"🐺 Beta:  Objectif={composite_objective(beta):.2f}")
        print(f"🐺 Delta: Objectif={composite_objective(delta):.2f}")

        # Historique
        best_history.append(alpha_obj)
        best_makespan_history.append(alpha.makespan)
        best_cost_history.append(alpha.cost)
        best_energy_history.append(alpha.energy)

        # Mise à jour de la population
        population = gwo_update_mono(valid_population, alpha, beta, delta, donnees, a)

    # Résultats finaux
    end_time = time.time()
    exec_time = end_time - start_time

    print("\n" + "="*80)
    print("RÉSULTATS FINAUX GWO MONO-OBJECTIF")
    print("="*80)
    
    final_valid = filter_by_lmax(population, donnees)
    if final_valid:
        best_solution = min(final_valid, key=composite_objective)
        lmax_valid, lmax_info = check_lmax_constraint(best_solution, donnees)
        
        print(f"\n🏆 MEILLEURE SOLUTION:")
        print(f"   Objectif composite: {composite_objective(best_solution):.4f}")
        print(f"   Makespan: {best_solution.makespan:.4f}")
        print(f"   Cost: {best_solution.cost:.4f}")
        print(f"   Energy: {best_solution.energy:.4f}")
        if lmax_info:
            print(f"   Contrainte Lmax: {'✓ Respectée' if lmax_valid else '✗ Violée'} "
                  f"(seuil: {lmax_info['threshold']:.4f})")
    
    print(f"\nTemps d'exécution: {exec_time:.2f} secondes")

    # Visualisation de la convergence
    plot_convergence(best_history, best_makespan_history, best_cost_history, 
                     best_energy_history)


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_convergence(best_history, makespan_hist, cost_hist, energy_hist):
    """Affiche la convergence de GWO mono-objectif."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Objectif composite
    ax = axes[0, 0]
    ax.plot(range(1, len(best_history) + 1), best_history, 
            marker='o', linewidth=2, color='blue')
    ax.set_xlabel('Itération', fontweight='bold')
    ax.set_ylabel('Objectif Composite', fontweight='bold')
    ax.set_title('Convergence - Objectif Composite (GWO)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Makespan
    ax = axes[0, 1]
    ax.plot(range(1, len(makespan_hist) + 1), makespan_hist, 
            marker='s', linewidth=2, color='green')
    ax.set_xlabel('Itération', fontweight='bold')
    ax.set_ylabel('Makespan', fontweight='bold')
    ax.set_title('Convergence - Makespan', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Cost
    ax = axes[1, 0]
    ax.plot(range(1, len(cost_hist) + 1), cost_hist, 
            marker='^', linewidth=2, color='orange')
    ax.set_xlabel('Itération', fontweight='bold')
    ax.set_ylabel('Cost', fontweight='bold')
    ax.set_title('Convergence - Cost', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Energy
    ax = axes[1, 1]
    ax.plot(range(1, len(energy_hist) + 1), energy_hist, 
            marker='d', linewidth=2, color='red')
    ax.set_xlabel('Itération', fontweight='bold')
    ax.set_ylabel('Energy', fontweight='bold')
    ax.set_title('Convergence - Energy', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('GWO Mono-Objectif - Évolution des Objectifs', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
