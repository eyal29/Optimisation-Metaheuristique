

#anciennement dans pareto.py:


# VISUALISATION 3D DES FRONTS
from matplotlib import pyplot as plt
import mplcursors
import numpy as np

from algo import generate_fronts
from metrics import _compute_composite_scores, compute_metrics_all_solutions, extract_objectives
from utils_to_algo import sol_signature


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


def plot_fronts_3d(valid_solutions, fronts, title="Fronts de Pareto (3D)", greedy_points=None):
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
    # AJOUT du tracé des points Gloutons
    if greedy_points:
        greedy_xs = [s.makespan for s in greedy_points.values()]
        greedy_ys = [s.cost for s in greedy_points.values()]
        greedy_zs = [s.energy for s in greedy_points.values()]
        
        # Assurez-vous d'avoir les noms dans le bon ordre pour les tooltips
        greedy_names = list(greedy_points.keys())
        
        sc_g = ax.scatter(greedy_xs, greedy_ys, greedy_zs, 
                          label="Solutions Gloutonnes",
                          s=150, marker='D', # Utilisation d'un losange (Diamond)
                          color='magenta', edgecolor='black', linewidth=1.5, alpha=1.0)
        
        # Tooltips pour les points Gloutons
        greedy_tooltips = []
        for name, sol in greedy_points.items():
             tooltip = (f"ALGO: {name}\n"
                       f"Makespan: {sol.makespan:.2f}\n"
                       f"Cost: {sol.cost:.2f}\n"
                       f"Energy: {sol.energy:.2f}")
             greedy_tooltips.append(tooltip)
        
        # Connecter le curseur aux points Gloutons
        cursor_g = mplcursors.cursor(sc_g, hover=True)
        cursor_g.connect("add", make_annotation(greedy_tooltips))
        
        all_scatters.append(sc_g)
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
    # plt.show()
    return fig




#anciennement dans metrics.py:

def plot_hv_convergence(hv_history, title="Convergence de l'hypervolume"):
    """Trace la courbe de convergence de l'hypervolume au fil des itérations."""
    if not hv_history:
        print("hv_history est vide, rien à tracer.")
        return None, None # Retourne None, None si rien à tracer

    fig = plt.figure(figsize=(8, 5)) # <-- Utiliser fig = plt.figure() pour retourner l'objet Figure
    ax = fig.add_subplot(111)
    
    ax.plot(range(len(hv_history)), hv_history, marker="o")
    ax.set_xlabel("Itération", fontsize=12)
    ax.set_ylabel("Hypervolume", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    # plt.show() <-- SUPPRIMÉ
    
    return fig, ax # <-- Retourne la Figure et les Axes

def plot_pareto_2d(objs, x_idx=0, y_idx=1, x_label="Makespan", y_label="Cost"):
    """
    Visualisation 2D de la Pareto front.
    La couleur représente l'énergie.
    """
    if objs is None or len(objs) == 0:
        print("objs est vide, rien à tracer.")
        return None, None # Retourne None, None si rien à tracer

    x = objs[:, x_idx]
    y = objs[:, y_idx]
    energy = objs[:, 2]

    fig = plt.figure(figsize=(7, 6)) # <-- Utiliser fig = plt.figure() pour retourner l'objet Figure
    ax = fig.add_subplot(111)
    
    sc = ax.scatter(x, y, c=energy, s=70, edgecolor="black", alpha=0.85)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title("Distribution des solutions Pareto (2D)", fontsize=14)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Energy", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    # plt.show() <-- SUPPRIMÉ
    
    return fig, ax # <-- Retourne la Figure et les Axes

#anciennement dans metrics.py graph pour l'archive:
def _create_metric_subplot(ax, solution_indices, values, metric_name, config):
    """
    Fonction auxiliaire pour créer un sous-graphique de métrique.
    
    Args:
        ax: axes matplotlib
        solution_indices: indices des solutions
        values: valeurs de la métrique
        metric_name: nom de la métrique
        config: dict avec 'color', 'edgecolor', 'title', 'ylabel', 'better', 'best_func'
    """
    ax.bar(solution_indices, values, color=config['color'], 
           alpha=0.7, edgecolor=config['edgecolor'])
    ax.set_xlabel('Solution', fontweight='bold')
    ax.set_ylabel(config['ylabel'], fontweight='bold')
    ax.set_title(config['title'], fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    mean_val = np.mean(values)
    ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'Moyenne: {mean_val:.3f}')
    
    best_idx = config['best_func'](values)
    best_val = values[best_idx]
    ax.text(0.98, 0.98, 
            f'{config["best_label"]}:\nSolution {best_idx + 1}\n{metric_name} = {best_val:.3f}',
            transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', 
                     alpha=0.95, edgecolor='green', linewidth=2))
    
    ax.text(0.02, 0.02, config['better'], 
            transform=ax.transAxes, fontsize=9, 
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightyellow', 
                     alpha=0.9, edgecolor='orange', linewidth=1.5))
    ax.legend(loc='upper left')


def plot_archive_metrics_visualization(metrics_data, archive_solutions):
    """
    Affiche les métriques de toutes les solutions de l'archive sous forme de graphiques.
    
    Args:
        metrics_data: dict contenant 'LBI', 'FUR', 'EE', 'AvgLatency'
        archive_solutions: liste des solutions de l'archive
    """
    if not metrics_data['LBI']:
        print("Aucune métrique à afficher.")
        return
    
    n_solutions = len(metrics_data['LBI'])
    solution_indices = np.arange(1, n_solutions + 1)
    
    fig = plt.figure(figsize=(18, 10))
    
    # Configuration des métriques
    metrics_config = {
        'LBI': {
            'color': 'steelblue', 'edgecolor': 'navy',
            'title': 'Load Balancing Index (LBI)', 'ylabel': 'LBI',
            'better': 'Plus bas = meilleur', 'best_func': np.argmin,
            'best_label': 'Meilleur équilibrage'
        },
        'FUR': {
            'color': 'seagreen', 'edgecolor': 'darkgreen',
            'title': 'Fog Utilization Ratio (FUR)', 'ylabel': 'FUR',
            'better': 'Plus haut = meilleur', 'best_func': np.argmax,
            'best_label': 'Max utilisation fog'
        },
        'EE': {
            'color': 'coral', 'edgecolor': 'darkred',
            'title': 'Energy Efficiency (EE)', 'ylabel': 'EE',
            'better': 'Plus haut = meilleur', 'best_func': np.argmax,
            'best_label': 'Plus efficace'
        },
        'AvgLatency': {
            'color': 'mediumpurple', 'edgecolor': 'indigo',
            'title': 'Average Latency', 'ylabel': 'Latence Moyenne',
            'better': 'Plus bas = meilleur', 'best_func': np.argmin,
            'best_label': 'Meilleure performance'
        }
    }
    
    # Créer les 4 graphiques de métriques
    for idx, (metric_name, config) in enumerate(metrics_config.items(), 1):
        ax = plt.subplot(2, 3, idx)
        _create_metric_subplot(ax, solution_indices, metrics_data[metric_name], 
                              metric_name, config)
    
    # Recommandations (subplot 5)
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    scores = _compute_composite_scores(metrics_data, n_solutions)
    top3_indices = np.argsort(scores)[-3:][::-1]
    
    recommendation_text = "RECOMMANDATIONS\n" + "="*50 + "\n"
    recommendation_text += "TOP 3 SOLUTIONS (score composite):\n"
    
    for rank, idx in enumerate(top3_indices, 1):
        sol_num = idx + 1
        # solution = archive_solutions[idx]
        recommendation_text += f"#{rank} - Solution {sol_num} (score: {scores[idx]:.3f})\n"
        # recommendation_text += f"   Makespan: {solution.makespan:.2f}\n"
        # recommendation_text += f"   Cost: {solution.cost:.2f}\n"
        # recommendation_text += f"   Energy: {solution.energy:.2f}\n"
        recommendation_text += f"   LBI: {metrics_data['LBI'][idx]:.3f} "
        recommendation_text += f"\n"
        recommendation_text += f"   FUR: {metrics_data['FUR'][idx]:.3f} "
        recommendation_text += f"\n"
        recommendation_text += f"   EE: {metrics_data['EE'][idx]:.1f} "
        recommendation_text += f"\n"
        recommendation_text += f"   Latency: {metrics_data['AvgLatency'][idx]:.3f} "
        recommendation_text += f"\n\n"
    
    ax5.text(0.05, 0.95, recommendation_text, transform=ax5.transAxes, 
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', 
                      alpha=0.8, edgecolor='blue', linewidth=2))
    
    # Statistiques (subplot 6)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = "STATISTIQUES DESCRIPTIVES\n" + "="*40 + "\n\n"
    
    for metric_name in ['LBI', 'FUR', 'EE', 'AvgLatency']:
        values = np.array(metrics_data[metric_name])
        stats_text += f"{metric_name}:\n"
        stats_text += f"  Moyenne: {np.mean(values):.4f}\n"
        stats_text += f"  Médiane: {np.median(values):.4f}\n"
        stats_text += f"  Écart-type: {np.std(values):.4f}\n"
        stats_text += f"  Min: {np.min(values):.4f}\n"
        stats_text += f"  Max: {np.max(values):.4f}\n\n"
    
    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', 
                      alpha=0.8, edgecolor='orange', linewidth=2))
    
    plt.suptitle(f'Analyse des Métriques - {n_solutions} Solutions de l\'Archive', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    # plt.show()
    return fig

#anciennement dans utils_main.py:

def plot_final_results(archive_unique, hv_history, donnees, valid_solutions=None, reference_solutions=None, metrics_data=None):
    """
    Génère tous les graphiques finaux avec affichage de l'archive complète.
    
    Args:
        archive_unique: Solutions uniques de l'archive (Front de Pareto final)
        hv_history: Historique de l'hypervolume
        donnees: Données du problème
        valid_solutions: Solutions valides (optionnel)
        greedy_points: Dictionnaire des solutions gloutonnes à superposer (AJOUTÉ)
    """
    print("\n" + "="*70)
    print("AFFICHAGE DE L'ARCHIVE GLOBALE (TOUS LES FRONTS)")
    print("="*70)
    
    archive_fronts = generate_fronts(archive_unique)
    figures = []
    if reference_solutions:
        greedy_points = {k: v for k, v in reference_solutions.items() if k.startswith('Greedy')}
        gwo_mono_points = {k: v for k, v in reference_solutions.items() if k.startswith('GWO-Mono')}
    else:
        greedy_points = {}
        gwo_mono_points = {}

    # 1. GRAPH 3D - GWO-NSGA-II vs GREEDY
    fig_3d_greedy = plot_fronts_3d(archive_unique, archive_fronts, 
                   title="Fronts de Pareto - GWO-NSGA-II vs Greedy",
                   greedy_points=greedy_points)
    if fig_3d_greedy: figures.append(fig_3d_greedy)
    
    # 2. GRAPH 3D - GWO-NSGA-II vs GWO MONO
    if gwo_mono_points:
        fig_3d_mono = plot_fronts_3d(archive_unique, archive_fronts, 
                       title="Fronts de Pareto - GWO-NSGA-II vs GWO Mono",
                       greedy_points=gwo_mono_points) # Réutiliser greedy_points car la structure est la même
        if fig_3d_mono: figures.append(fig_3d_mono)


    # 3. Plot HV
    if hv_history:
        fig_hv, _ = plot_hv_convergence(hv_history, title="Convergence de l'hypervolume (GWO Fog-Cloud)")
        if fig_hv: figures.append(fig_hv)

    # 4. Plot 2D
    if archive_unique:
        objs_arch_final = extract_objectives(archive_unique)
        fig_2d, _ = plot_pareto_2d(objs_arch_final, x_idx=0, y_idx=1, 
                      x_label="Makespan", y_label="Cost")
        if fig_2d: figures.append(fig_2d)
        
    # 5. Plot Métriques d'Archive
    if metrics_data:
        # Note: plot_archive_metrics_visualization a été modifié en 1.1
        fig_metrics = plot_archive_metrics_visualization(metrics_data, archive_unique)
        if fig_metrics: figures.append(fig_metrics)
        
    print(f"✓ {len(figures)} figures générées et affichées simultanément.")
    
    # AFFICHAGE SIMULTANÉ DE TOUTES LES FIGURES
    plt.show()
    # return