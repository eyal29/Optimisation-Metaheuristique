import matplotlib.pyplot as plt
import numpy as np
from algo_utils import assign_crowding_distance, crowding_distance
import mplcursors
import copy

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
            fronts = generate_fronts(self.archive, return_indices=True)

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

def sol_signature(sol):
    """Crée une signature unique pour une solution basée sur ses objectifs.
    
    Utilisé pour comparer les solutions par valeurs plutôt que par identité d'objet,
    notamment après des copies profondes.
    
    Args:
        sol: Solution avec attributs makespan, cost, energy
        
    Returns:
        tuple: Signature (makespan, cost, energy) arrondie à 4 décimales
    """
    return (round(sol.makespan, 4), round(sol.cost, 4), round(sol.energy, 4))

def generate_fronts(population, return_indices: bool = False):
    """
    Calcule les fronts de Pareto.

    Parameters
    ----------
    population : list[Solution]
        Liste de solutions.
    return_indices : bool
        - False (par défaut) : retourne des listes de solutions [[sol, ...], ...]
        - True : retourne des listes d'indices [[i1, i2, ...], [j1, ...], ...]

    Returns
    -------
    fronts : list[list]
        Liste de fronts, chaque front est une liste de solutions OU d'indices.
    """
    remaining = list(range(len(population)))  # on travaille sur des indices
    fronts_idx = []

    while remaining:
        current_front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                # on réutilise la fonction dominates(a, b)
                if dominates(population[j], population[i]):
                    dominated = True
                    break
            if not dominated:
                current_front.append(i)

        fronts_idx.append(current_front)
        remaining = [i for i in remaining if i not in current_front]

    if return_indices:
        return fronts_idx

    # sinon on renvoie les solutions correspondantes
    return [[population[i] for i in front] for front in fronts_idx]


# les leaders
def select_leaders(population):
    """
    Retourne (alpha, beta, delta) en respectant :
    - d'abord le rang de front (F1 > F2 > F3)
    - à l'intérieur de F1 : on choisit les solutions les plus diversifiées
      (crowding distance élevée)
    """

    if not population:
        return None, None, None

    # Fronts Pareto sur la population courante
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

#Affichage graphique 
def plot_fronts_3d(valid_solutions, fronts, title="Fronts de Pareto (3D)"):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    # Palette de couleurs distinctes pour chaque front
    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "brown", "pink", "olive"]
    colormaps = ['plasma', 'viridis', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter']

    # réglages visuels
    ax.set_facecolor("white")
    ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)

    all_scatters = []
    all_tooltips = []
    surf_handles = []
    surf_labels = []
    
    # Créer un mapping de solution (par valeurs) vers son index dans valid_solutions
    sol_to_idx = {sol_signature(sol): idx + 1 for idx, sol in enumerate(valid_solutions)}

    for i, front in enumerate(fronts):
        if not front:  # Skip empty fronts
            continue
            
        xs, ys, zs = [], [], []
        tooltips = []

        for solution in front:
            xs.append(solution.makespan)
            ys.append(solution.cost)
            zs.append(solution.energy)
            # Tooltip avec les métriques + numéro du front + vrai numéro de solution
            sol_num = sol_to_idx.get(sol_signature(solution), "?")
            tooltip = (f"Front {i+1} - Solution {sol_num}\n"
                      f"Makespan: {solution.makespan:.2f}\n"
                      f"Cost: {solution.cost:.2f}\n"
                      f"Energy: {solution.energy:.2f}")
            tooltips.append(tooltip)

        # Si beaucoup de points se superposent, ajouter un léger jitter pour les séparer visuellement
        unique_pts = len(set(zip(xs, ys, zs)))
        if unique_pts < len(xs):
            # calculer une échelle de jitter proportionnelle à l'étendue des données
            range_x = max(xs) - min(xs) if max(xs) != min(xs) else 1.0
            range_y = max(ys) - min(ys) if max(ys) != min(ys) else 1.0
            range_z = max(zs) - min(zs) if max(zs) != min(zs) else 1.0
            jitter_scale = 0.03  # 3% de l'étendue pour bien séparer
            xs = [x + np.random.normal(scale=jitter_scale * range_x) for x in xs]
            ys = [y + np.random.normal(scale=jitter_scale * range_y) for y in ys]
            zs = [z + np.random.normal(scale=jitter_scale * range_z) for z in zs]

        color = colors[i % len(colors)]
        cmap_name = colormaps[i % len(colormaps)]

        # Tous les fronts : tracer une surface lissée (style heatmap) + points
        if len(xs) >= 3:
            tri = ax.plot_trisurf(xs, ys, zs, cmap=cmap_name, alpha=0.55, linewidth=0.25, edgecolor='k')
            surf_handles.append(tri)
            surf_labels.append(f"Front {i+1} surface")
            # Points du front par-dessus en blanc
            sc = ax.scatter(xs, ys, zs, label=f"Front {i+1} ({len(front)} sols)",
                           s=55, color='white', edgecolor='black', linewidth=0.8, alpha=0.9)
        else:
            # Si moins de 3 points, juste scatter
            sc = ax.scatter(xs, ys, zs, label=f"Front {i+1} ({len(front)} sols)",
                           s=55, color=color, edgecolor='black', linewidth=0.8, alpha=0.8)
        
        all_scatters.append(sc)
        all_tooltips.append(tooltips)

        # Ligne reliant les points du front 1 uniquement
        if i == 0 and len(xs) > 1:
            sorted_points = sorted(zip(xs, ys, zs), key=lambda t: t[0])
            lx = [p[0] for p in sorted_points]
            ly = [p[1] for p in sorted_points]
            lz = [p[2] for p in sorted_points]
            ax.plot(lx, ly, lz, linestyle='-', linewidth=2.0, color='black', alpha=0.6)

    # Ajouter le curseur interactif pour tous les scatters
    for sc, tooltips in zip(all_scatters, all_tooltips):
        cursor = mplcursors.cursor(sc, hover=True)
        
        # Utiliser une closure pour capturer tooltips
        def make_annotation(tooltip_list):
            def on_add(sel):
                idx_point = sel.index
                if idx_point < len(tooltip_list):
                    sel.annotation.set_text(tooltip_list[idx_point])
                    sel.annotation.get_bbox_patch().set(alpha=0.95, facecolor='lightyellow', 
                                                        edgecolor='black', linewidth=1.5)
                    sel.annotation.set_fontsize(10)
            return on_add
        
        cursor.connect("add", make_annotation(tooltips))

    # Axes + titres
    ax.set_xlabel("Makespan", fontsize=13, labelpad=15, fontweight='bold')
    ax.set_ylabel("Cost", fontsize=13, labelpad=15, fontweight='bold')
    ax.set_zlabel("Energy", fontsize=13, labelpad=15, fontweight='bold')
    ax.set_title(title, fontsize=17, pad=20, fontweight='bold')

    # Désactiver la perspective pour avoir un repère orthogonal
    ax.set_proj_type('ortho')

    ax.view_init(elev=20, azim=35)
    plt.legend(fontsize=11, loc='upper left')

    # Ajouter colorbar si on a une surface
    if surf_handles:
        mappable = surf_handles[0]
        plt.colorbar(mappable, shrink=0.6, aspect=12, pad=0.08, label='Energy (surface)')
    plt.tight_layout()
    plt.show()

