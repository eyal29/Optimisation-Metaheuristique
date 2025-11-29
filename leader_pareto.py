import matplotlib.pyplot as plt
import mplcursors

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
    colors = ["red", "blue", "green", "orange", "purple"]  # Couleurs par front

    # réglages visuels
    ax.set_facecolor("white")
    ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)

    for i, front in enumerate(fronts):
        xs, ys, zs = [], [], []
        tooltips = []   # texte à afficher au survol

        for solution in front:
            xs.append(solution.makespan)
            ys.append(solution.cost)
            zs.append(solution.energy)
            tooltips.append(f"Affectation : {solution.assignment}") #texte quand on survoles le point 

        color = colors[i % len(colors)]

        # On dessine les points
        sc = ax.scatter( xs, ys, zs, label=f"Front {i+1}", s=80, color=color, edgecolor="black", alpha=0.85)
        cursor = mplcursors.cursor(sc, hover=True) # Survol interactif 

        @cursor.connect("add")
        def on_add(sel, texts=tooltips):
            idx_point = sel.index
            sel.annotation.set_text(texts[idx_point])  # affiche "Affectation : [...]"
            sel.annotation.get_bbox_patch().set(alpha=0.9)

        # Ligne reliant les points du front 1
        if i == 0 and len(xs) > 1:
            sorted_points = sorted(zip(xs, ys, zs), key=lambda t: t[0])
            lx = [p[0] for p in sorted_points]
            ly = [p[1] for p in sorted_points]
            lz = [p[2] for p in sorted_points]
            ax.plot(
                lx, ly, lz,
                linestyle='-',
                linewidth=2.5,
                color=color,
                alpha=0.8
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
def select_leaders(population):
    """
    Retourne (alpha, beta, delta) avec :
      - alpha : meilleure solution du front 1
      - beta  : solution de F1 la plus éloignée de alpha (objectifs différents)
      - delta : solution (dans toute la population) éloignée en makespan/cost et avec objectifs différents de alpha et beta
    """
    if not population:
        return None, None, None

    fronts = generate_fronts(population)
    front1 = fronts[0]
    def score(s):
        return s.makespan + s.cost + s.energy # qualité globale (plus petit = meilleur)

    def obj_key(s, ndigits=10):
        # clé pour comparer les objectifs (on arrondit un peu pour éviter les micro-différences floats)
        return (
            round(s.makespan, ndigits),
            round(s.cost, ndigits),
            round(s.energy, ndigits),
        )

    def dist_3d(a, b):
        # distance euclidienne dans l'espace (makespan, cost, energy)
        return (
            (a.makespan - b.makespan) ** 2
            + (a.cost     - b.cost)   ** 2
            + (a.energy   - b.energy) ** 2
        ) ** 0.5

    # --- 1) Alpha : meilleure solution du front 1 ---
    alpha = min(front1, key=score)
    key_alpha = obj_key(alpha)

    # --- 2) Beta : solution de F1 la plus éloignée de alpha, avec objectifs différents ---
    beta = None
    key_beta = None

    beta_candidates = [
        s for s in front1
        if obj_key(s) != key_alpha
    ]

    if beta_candidates:
        beta = max(beta_candidates, key=lambda s: dist_3d(s, alpha))
        key_beta = obj_key(beta)

    # --- 3) Delta : solution éloignée en makespan/cost, objectifs différents de alpha et beta ---

    delta = None
    if beta is not None:
        delta_candidates = [
            s for s in population
            if obj_key(s) != key_alpha and obj_key(s) != key_beta
        ]
    else:
        # si on n'a pas pu trouver de beta distinct, on évite juste d'égaliser alpha
        delta_candidates = [
            s for s in population
            if obj_key(s) != key_alpha
        ]

    if delta_candidates:
        delta = max(
            delta_candidates,
            key=lambda s: abs(s.makespan - alpha.makespan)
                        + abs(s.cost     - alpha.cost)
        )

    return alpha, beta, delta
