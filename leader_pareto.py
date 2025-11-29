import matplotlib.pyplot as plt
import numpy as np
from gwo_utils import crowding_distance


def dominates(a, b): # permet de savoir si a domine b 
    """Retourne True si a domine b (minimisation)."""
    return (
        (a.makespan <= b.makespan) and
        (a.cost <= b.cost) and
        (a.energy <= b.energy) and
        (
            a.makespan < b.makespan or
            a.cost < b.cost or
            a.energy < b.energy
        )
    )

def  generate_fronts(population):
    """Retourne TOUS les fronts non dominés : """
    remaining = population.copy()
    fronts = []
    while len(remaining) > 0:
        current_front = [] #crée un nouveau front vide

        # Trouver les solutions non dominées DANS remaining
        for s in remaining:
            dominated = False
            for other in remaining:
                if other is s:
                    continue
                if dominates(other, s):
                    dominated = True
                    break
            if not dominated:
                current_front.append(s)

        fronts.append(current_front) # Ajouter ce front

        remaining = [s for s in remaining if s not in current_front]  # Retirer ces solutions de la liste restante
    return fronts

#Affichage graphique 
def plot_fronts_3d(valid_solutions, fronts):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ["red", "blue", "green", "orange", "purple"]   # Couleurs par front

    #reglage pour que ca soit plus facile a comprendre 
    ax.set_facecolor("white")
    ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)

    for i, front in enumerate(fronts):
        xs, ys, zs = [], [], []
        labels = []

        for sol in front:
            idx = valid_solutions.index(sol) + 1
            xs.append(sol.makespan)
            ys.append(sol.cost)
            zs.append(sol.energy)
            #labels.append(f"S{idx}")
        color = colors[i % len(colors)]

        # Points  gros et visibles
        ax.scatter(xs, ys, zs, label=f"Front {i+1}", s=80, color=color, edgecolor="black", alpha=0.85)
        # Étiquette de chaque point
        for x, y, z, lab in zip(xs, ys, zs, labels):
            ax.text(
                    x, y, z, lab,
                    fontsize=12, fontweight="bold",
                    color="black", ha="center", va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc="white",             # fond bien visible
                        ec="black",             # contour noir plus net
                        lw=1.2,                 # contour plus épais
                        alpha=0.95              # haute visibilité
                    ))
            
        # Ligne du front 1
        if i == 0 and len(xs) > 1:
            sorted_points = sorted(zip(xs, ys, zs), key=lambda t: t[0])
            lx = [p[0] for p in sorted_points]
            ly = [p[1] for p in sorted_points]
            lz = [p[2] for p in sorted_points]
            ax.plot(
                lx, ly, lz,
                linestyle='-', linewidth=2.5,
                color=color, alpha=0.8
            )

    # Axes + titres
    ax.set_xlabel("Makespan", fontsize=12, labelpad=15)
    ax.set_ylabel("Cost", fontsize=12, labelpad=15)
    ax.set_zlabel("Energy", fontsize=12, labelpad=15)
    ax.set_title("Fronts de Pareto (3D)", fontsize=16, pad=20)
    ax.view_init(elev=20, azim=35)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# les leaders
# les leaders
def select_leaders(population):
    """
    Retourne (alpha, beta, delta) en respectant :
    - d'abord le rang de front (F1 > F2 > F3)
    - à l'intérieur de F1 : on choisit les solutions les plus diversifiées
      (crowding distance élevée).
    """

    if not population:
        return None, None, None

    # 1) Fronts Pareto sur la population courante
    fronts = generate_fronts(population)

    if len(fronts) == 0 or len(fronts[0]) == 0:
        return None, None, None

    F1 = fronts[0]   # front 1 = solutions non dominées

    # --- Cas normal : on a au moins 3 solutions dans F1 ---
    if len(F1) >= 3:
        # Matrice des objectifs pour F1 : shape (N, 3)
        front_objs = np.array([
            [s.makespan, s.cost, s.energy]
            for s in F1
        ], dtype=float)

        # Crowding distance sur le front 1
        cd = crowding_distance(front_objs)

        # Indices triés par crowding décroissante
        sorted_idx = np.argsort(-cd)

        alpha = F1[sorted_idx[0]]
        beta  = F1[sorted_idx[1]]
        delta = F1[sorted_idx[2]]
        return alpha, beta, delta

    # --- Cas rare : F1 contient moins de 3 solutions ---
    # On revient à une logique de fallback proche de ton ancienne version :
    def score(s):
        return s.makespan + s.cost + s.energy

    leaders = []
    for front in fronts:
        sorted_front = sorted(front, key=score)
        for s in sorted_front:
            leaders.append(s)
            if len(leaders) == 3:
                return leaders[0], leaders[1], leaders[2]

    # Si vraiment on a moins de 3 solutions au total
    while len(leaders) < 3:
        leaders.append(None)

    return leaders[0], leaders[1], leaders[2]
