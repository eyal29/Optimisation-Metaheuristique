import copy
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from algo_utils import assign_crowding_distance, crowding_distance

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
        print(" ➤ Tentative d'ajout d'une nouvelle solution :")
        print(f"     Taille archive AVANT ajout : {len(self.archive)}")

        # Si sol est dominée par une solution de l’archive → on la rejette
        for s in self.archive:
            if self._dominates(s, sol):
                print("   ✘ SOLUTION REJETÉE : elle est dominée par une solution de l'archive.")
                print(f"     Taille archive APRÈS tentative : {len(self.archive)}")
                return False

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

# FONCTIONS DE DOMINANCE ET SIGNATURE
def dominates(a, b):
    return (
        a.makespan <= b.makespan and
        a.cost <= b.cost and
        a.energy <= b.energy and
        (a.makespan < b.makespan or a.cost < b.cost or a.energy < b.energy)
    )

def sol_signature(sol):
    return (round(sol.makespan, 4), round(sol.cost, 4), round(sol.energy, 4))


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


# VISUALISATION 3D DES FRONTS
def _add_jitter_if_needed(xs, ys, zs):
    """Ajoute un léger jitter si des points se superposent."""
    unique_pts = len(set(zip(xs, ys, zs)))
    if unique_pts >= len(xs):
        return xs, ys, zs
    
    # Calculer échelle de jitter proportionnelle à l'étendue
    range_x = max(xs) - min(xs) if max(xs) != min(xs) else 1.0
    range_y = max(ys) - min(ys) if max(ys) != min(ys) else 1.0
    range_z = max(zs) - min(zs) if max(zs) != min(zs) else 1.0
    jitter_scale = 0.03  # 3% de l'étendue
    
    xs = [x + np.random.normal(scale=jitter_scale * range_x) for x in xs]
    ys = [y + np.random.normal(scale=jitter_scale * range_y) for y in ys]
    zs = [z + np.random.normal(scale=jitter_scale * range_z) for z in zs]
    
    return xs, ys, zs


def _create_tooltips(front, front_idx, sol_to_idx):
    """Crée les tooltips pour les solutions d'un front."""
    tooltips = []
    for solution in front:
        sol_num = sol_to_idx.get(sol_signature(solution), "?")
        tooltip = (f"Front {front_idx} - Solution {sol_num}\n"
                  f"Makespan: {solution.makespan:.2f}\n"
                  f"Cost: {solution.cost:.2f}\n"
                  f"Energy: {solution.energy:.2f}")
        tooltips.append(tooltip)
    return tooltips


def _plot_front(ax, front, front_idx, sol_to_idx, colors, colormaps):
    """Trace un front de Pareto en 3D avec surface et points."""
    xs = [s.makespan for s in front]
    ys = [s.cost for s in front]
    zs = [s.energy for s in front]
    
    tooltips = _create_tooltips(front, front_idx, sol_to_idx)
    xs, ys, zs = _add_jitter_if_needed(xs, ys, zs)
    
    color = colors[(front_idx - 1) % len(colors)]
    cmap_name = colormaps[(front_idx - 1) % len(colormaps)]
    
    surf = None
    # Tracer surface si au moins 3 points
    if len(xs) >= 3:
        surf = ax.plot_trisurf(xs, ys, zs, cmap=cmap_name, alpha=0.55, 
                               linewidth=0.25, edgecolor='k')
        sc = ax.scatter(xs, ys, zs, label=f"Front {front_idx} ({len(front)} sols)",
                       s=55, color='white', edgecolor='black', linewidth=0.8, alpha=0.9)
    else:
        sc = ax.scatter(xs, ys, zs, label=f"Front {front_idx} ({len(front)} sols)",
                       s=55, color=color, edgecolor='black', linewidth=0.8, alpha=0.8)
    
    # Ligne reliant les points du front 1
    if front_idx == 1 and len(xs) > 1:
        sorted_points = sorted(zip(xs, ys, zs), key=lambda t: t[0])
        lx, ly, lz = zip(*sorted_points)
        ax.plot(lx, ly, lz, linestyle='-', linewidth=2.0, color='black', alpha=0.6)
    
    return sc, tooltips, surf


def plot_fronts_3d(valid_solutions, fronts, title="Fronts de Pareto (3D)"):
    """Affiche les fronts de Pareto en 3D avec surfaces interpolées et tooltips interactifs.
    
    Args:
        valid_solutions: Liste complète des solutions
        fronts: Liste des fronts de Pareto
        title: Titre du graphique
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Palettes de couleurs
    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "brown", "pink", "olive"]
    colormaps = ['plasma', 'viridis', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter']
    
    # Configuration visuelle
    ax.set_facecolor("white")
    ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)
    
    # Mapping solution -> index
    sol_to_idx = {sol_signature(sol): idx + 1 for idx, sol in enumerate(valid_solutions)}
    
    all_scatters = []
    all_tooltips = []
    surf_handles = []
    
    # Tracer chaque front
    for i, front in enumerate(fronts, 1):
        if not front:
            continue
        
        sc, tooltips, surf = _plot_front(ax, front, i, sol_to_idx, colors, colormaps)
        all_scatters.append(sc)
        all_tooltips.append(tooltips)
        if surf:
            surf_handles.append(surf)
    
    # Ajouter curseurs interactifs
    for sc, tooltips in zip(all_scatters, all_tooltips):
        cursor = mplcursors.cursor(sc, hover=True)
        
        def make_annotation(tooltip_list):
            def on_add(sel):
                if sel.index < len(tooltip_list):
                    sel.annotation.set_text(tooltip_list[sel.index])
                    sel.annotation.get_bbox_patch().set(
                        alpha=0.95, facecolor='lightyellow', 
                        edgecolor='black', linewidth=1.5
                    )
                    sel.annotation.set_fontsize(10)
            return on_add
        
        cursor.connect("add", make_annotation(tooltips))
    
    # Configuration des axes
    ax.set_xlabel("Makespan", fontsize=13, labelpad=15, fontweight='bold')
    ax.set_ylabel("Cost", fontsize=13, labelpad=15, fontweight='bold')
    ax.set_zlabel("Energy", fontsize=13, labelpad=15, fontweight='bold')
    ax.set_title(title, fontsize=17, pad=20, fontweight='bold')
    ax.set_proj_type('ortho')
    ax.view_init(elev=20, azim=35)
    
    plt.legend(fontsize=11, loc='upper left')
    
    if surf_handles:
        plt.colorbar(surf_handles[0], shrink=0.6, aspect=12, pad=0.08, label='Energy (surface)')
    
    plt.tight_layout()
    plt.show()

