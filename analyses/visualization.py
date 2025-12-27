from matplotlib import pyplot as plt
import mplcursors
import numpy as np

from algo_hybride.algo import generate_fronts
from analyses.metrics import compute_composite_scores, extract_objectives
from algo_hybride.utils_to_algo import sol_signature


def _add_jitter_if_needed(xs, ys, zs):
    """Ajoute un léger jitter si des points se superposent."""
    unique_pts = len(set(zip(xs, ys, zs)))
    if unique_pts >= len(xs):
        return xs, ys, zs
    
    range_x = max(xs) - min(xs) if max(xs) != min(xs) else 1.0
    range_y = max(ys) - min(ys) if max(ys) != min(ys) else 1.0
    range_z = max(zs) - min(zs) if max(zs) != min(zs) else 1.0
    jitter_scale = 0.03  # 3% de l'étendue
    
    xs = [x + np.random.normal(scale=jitter_scale * range_x) for x in xs]
    ys = [y + np.random.normal(scale=jitter_scale * range_y) for y in ys]
    zs = [z + np.random.normal(scale=jitter_scale * range_z) for z in zs]
    
    return xs, ys, zs


def _create_tooltips(front, front_idx, sol_to_idx):
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
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "brown", "pink", "olive"]
    colormaps = ['plasma', 'viridis', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter']
    
    ax.set_facecolor("white")
    ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)
    
    sol_to_idx = {sol_signature(sol): idx + 1 for idx, sol in enumerate(valid_solutions)}
    
    all_scatters = []
    all_tooltips = []
    surf_handles = []
    
    for i, front in enumerate(fronts, 1):
        if not front:
            continue
        
        sc, tooltips, surf = _plot_front(ax, front, i, sol_to_idx, colors, colormaps)
        all_scatters.append(sc)
        all_tooltips.append(tooltips)
        if surf:
            surf_handles.append(surf)
    
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

    if greedy_points:
        
        single_points_xs = []
        single_points_ys = []
        single_points_zs = []
        single_points_names = []
        
        nsga2_front_xs = []
        nsga2_front_ys = []
        nsga2_front_zs = []
        nsga2_front_tooltips = []
        
        for name, sol_or_list in greedy_points.items():
            if isinstance(sol_or_list, list):
                # Ceci est le Front NSGA-II Simple (liste de solutions)
                for sol in sol_or_list:
                    nsga2_front_xs.append(sol.makespan)
                    nsga2_front_ys.append(sol.cost)
                    nsga2_front_zs.append(sol.energy)
                    tooltip = (f"ALGO: {name}\n"
                              f"Makespan: {sol.makespan:.2f}\n"
                              f"Cost: {sol.cost:.2f}\n"
                              f"Energy: {sol.energy:.2f}")
                    nsga2_front_tooltips.append(tooltip)
            else:
                # Solution unique (Greedy ou GWO Mono)
                single_points_xs.append(sol_or_list.makespan)
                single_points_ys.append(sol_or_list.cost)
                single_points_zs.append(sol_or_list.energy)
                single_points_names.append(name)
                
        # 💡 TRACÉ DU FRONT NSGA-II (Points 'D', sans surface, couleur distincte)
        if nsga2_front_xs:
            sc_nsga2 = ax.scatter(nsga2_front_xs, nsga2_front_ys, nsga2_front_zs, 
                                  label="Front NSGA-II Simple",
                                  s=80, marker='D', 
                                  color='cyan', edgecolor='black', linewidth=1.5, alpha=0.9)
            
            cursor_nsga2 = mplcursors.cursor(sc_nsga2, hover=True)
            cursor_nsga2.connect("add", make_annotation(nsga2_front_tooltips))
            all_scatters.append(sc_nsga2)
            
            # Ligne pour mieux visualiser la continuité du Front
            sorted_points = sorted(zip(nsga2_front_xs, nsga2_front_ys, nsga2_front_zs), key=lambda t: t[0])
            lx, ly, lz = zip(*sorted_points)
            ax.plot(lx, ly, lz, linestyle='--', linewidth=1.5, color='cyan', alpha=0.7)


        # TRACÉ DES POINTS UNIQUES (Solutions Adversaires)
        if single_points_xs:
            greedy_tooltips = []
            for name, sol in zip(single_points_names, [greedy_points[n] for n in single_points_names]):
                 tooltip = (f"ALGO: {name}\n"
                           f"Makespan: {sol.makespan:.2f}\n"
                           f"Cost: {sol.cost:.2f}\n"
                           f"Energy: {sol.energy:.2f}")
                 greedy_tooltips.append(tooltip)
            
            sc_g = ax.scatter(single_points_xs, single_points_ys, single_points_zs, 
                              label="Solutions Adversaires (Mono/Greedy)",
                              s=150, marker='s', # Utilisation d'un carré pour distinguer
                              color='magenta', edgecolor='black', linewidth=1.5, alpha=1.0)
            
            cursor_g = mplcursors.cursor(sc_g, hover=True)
            cursor_g.connect("add", make_annotation(greedy_tooltips))
            
            all_scatters.append(sc_g)

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
    return fig


def plot_hv_convergence(hv_history, title="Convergence de l'hypervolume"):
    if not hv_history:
        print("hv_history est vide, rien à tracer.")
        return None, None 

    fig = plt.figure(figsize=(8, 5)) 
    ax = fig.add_subplot(111)
    
    ax.plot(range(len(hv_history)), hv_history, marker="o")
    ax.set_xlabel("Itération", fontsize=12)
    ax.set_ylabel("Hypervolume", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig, ax 

def plot_hv_comparison(hv1, hv2, label1="GWO-NSGA-II", label2="NSGA-II Simple", title="Comparaison de la Convergence de l'Hypervolume"):
    """
    Trace et compare les historiques d'Hypervolume de deux algorithmes.
    """
    if not hv1 or not hv2:
        print("Historique HV incomplet pour la comparaison, rien à tracer.")
        return None
    
    # S'assurer que les listes ont la même longueur pour le tracé (en coupant au plus court)
    min_len = min(len(hv1), len(hv2))
    hv1 = hv1[:min_len]
    hv2 = hv2[:min_len]
    
    fig = plt.figure(figsize=(9, 6)) 
    ax = fig.add_subplot(111)
    
    ax.plot(range(min_len), hv1, marker="o", linestyle="-", 
            label=label1, color="red")
    ax.plot(range(min_len), hv2, marker="x", linestyle="--", 
            label=label2, color="blue")
            
    ax.set_xlabel("Itération", fontsize=12)
    ax.set_ylabel("Hypervolume (HV)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    fig.tight_layout()
    return fig

def plot_pareto_2d(objs, x_idx=0, y_idx=1, x_label="Makespan", y_label="Cost"):
    """
    Visualisation 2D de la Pareto front.
    La couleur représente l'énergie.
    """
    if objs is None or len(objs) == 0:
        print("objs est vide, rien à tracer.")
        return None, None 

    x = objs[:, x_idx]
    y = objs[:, y_idx]
    energy = objs[:, 2]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    
    sc = ax.scatter(x, y, c=energy, s=70, edgecolor="black", alpha=0.85)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title("Distribution des solutions Pareto (2D)", fontsize=14)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Energy", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()    
    return fig, ax

def _create_metric_subplot(ax, solution_indices, values, metric_name, config):
    """
    Fonction auxiliaire pour créer un sous-graphique de métrique.
    """
    values = np.array(values) 
    
    ax.bar(solution_indices, values, color=config['color'], 
           alpha=0.7, edgecolor=config['edgecolor'])
    ax.set_xlabel('Solution', fontweight='bold')
    ax.set_ylabel(config['ylabel'], fontweight='bold')
    ax.set_title(config['title'], fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    mean_val = np.mean(values)
    ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'Moyenne: {mean_val:.3f}')
    
    # 2. CORRECTION DE LA VÉRIFICATION (.size > 0) et de np.max/np.min
    if values.size > 0: 
        y_max_data = max(np.max(values), mean_val)
        y_min_data = min(np.min(values), 0)
    else:
        y_max_data = 0
        y_min_data = 0
        
    y_range = y_max_data - y_min_data if y_max_data != y_min_data else 1
    
    # Ajout d'un padding de 20% à la limite supérieure pour éviter le chevauchement
    padding_factor = 0.20 
    ax.set_ylim(y_min_data, y_max_data + y_range * padding_factor) 

    
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


def plot_final_results(archive_unique, hv_history, donnees, valid_solutions=None, reference_solutions=None, metrics_data=None):
    print("\n" + "="*70)
    print("AFFICHAGE DE L'ARCHIVE GLOBALE (TOUS LES FRONTS)")
    print("="*70)
    
    archive_fronts = generate_fronts(archive_unique)
    figures = []
    greedy_points = {}
    gwo_mono_points = {}
    nsga2_front_ref = {}
    mogwo_front_ref = {} # 💡 Initialisation pour MOGWO
    nsga2_hv_history = None
    mogwo_hv_history = None

    if reference_solutions:
        for k, v in reference_solutions.items():
            if k == 'NSGA2-Simple-HV-History': 
                nsga2_hv_history = v
            elif k == 'MOGWO-Standard-HV-History':
                mogwo_hv_history = v
            elif k.startswith('Greedy'):
                greedy_points[k] = v
            elif k.startswith('GWO-Mono'):
                gwo_mono_points[k] = v
            elif k.startswith('NSGA2-Simple'): 
                nsga2_front_ref[k] = v
            elif k == 'MOGWO-Standard-Archive':
                mogwo_front_ref[k] = v # 💡 Stockage pour usage général
            
    combined_single_points = {**greedy_points, **gwo_mono_points}
    # 1. GRAPH 3D - GWO-NSGA-II vs GREEDY
    fig_3d_greedy = plot_fronts_3d(archive_unique, archive_fronts, 
                   title="Fronts de Pareto - GWO-NSGA-II vs Greedy",
                   greedy_points=greedy_points)
    if fig_3d_greedy: figures.append(fig_3d_greedy)
    
    # 2. GRAPH 3D - GWO-NSGA-II vs GWO MONO
    if gwo_mono_points:
        fig_3d_mono = plot_fronts_3d(archive_unique, archive_fronts, 
                       title="Fronts de Pareto - GWO-NSGA-II vs GWO Mono",
                       greedy_points=gwo_mono_points) 
        if fig_3d_mono: figures.append(fig_3d_mono)

    if nsga2_front_ref:
        fig_3d_nsga2_simple = plot_fronts_3d(archive_unique, archive_fronts, 
                       title="Fronts de Pareto - GWO-NSGA-II vs NSGA-II Simple Front",
                       greedy_points=nsga2_front_ref) 
        if fig_3d_nsga2_simple: figures.append(fig_3d_nsga2_simple)

    # 3. GRAPH 3D - GWO-NSGA-II vs MOGWO (General Plot)
    if mogwo_front_ref:
        fig_3d_mogwo = plot_fronts_3d(archive_unique, archive_fronts, 
                       title="Fronts de Pareto - GWO-NSGA-II vs MOGWO Standard",
                       greedy_points=mogwo_front_ref) 
        if fig_3d_mogwo: figures.append(fig_3d_mogwo)
    # 3. Plot HV
    if hv_history:
        fig_hv, _ = plot_hv_convergence(hv_history, title="Convergence de l'hypervolume (GWO Fog-Cloud)")
        if fig_hv: figures.append(fig_hv)

    if hv_history and nsga2_hv_history:
        fig_hv_comp = plot_hv_comparison(hv_history, nsga2_hv_history, 
                                         label1="GWO-NSGA-II", label2="NSGA-II Simple",
                                         title="Comparaison HV: GWO-NSGA-II vs NSGA-II")
        if fig_hv_comp: figures.append(fig_hv_comp)

    if hv_history and mogwo_hv_history:
        fig_hv_comp_mogwo = plot_hv_comparison(hv_history, mogwo_hv_history,
                                               label1="GWO-NSGA-II", label2="MOGWO Standard",
                                               title="Comparaison HV: GWO-NSGA-II vs MOGWO")
        if fig_hv_comp_mogwo: figures.append(fig_hv_comp_mogwo)

    # 4. Plot 2D
    if archive_unique:
        objs_arch_final = extract_objectives(archive_unique)
        fig_2d, _ = plot_pareto_2d(objs_arch_final, x_idx=0, y_idx=1, 
                      x_label="Makespan", y_label="Cost")
        if fig_2d: figures.append(fig_2d)
        
    # 5. Plot Métriques d'Archive
    if metrics_data:
        n_solutions = len(archive_unique)
        fig_metrics_bars = plot_archive_metric_bars(metrics_data, n_solutions, 
                                title=f"Visualisation des Métriques (Barres) - {n_solutions} solutions")
        if fig_metrics_bars: figures.append(fig_metrics_bars)
        
        fig_metrics_summary = plot_archive_text_summary(metrics_data, archive_unique, 
                            title=f"Analyse des Métriques - Résumé ({n_solutions} solutions)")
        if fig_metrics_summary: figures.append(fig_metrics_summary)
        
    print(f"✓ {len(figures)} figures générées et affichées simultanément.")
    
    plt.show()

def plot_archive_metric_bars(metrics_data, n_solutions, title="Visualisation des Métriques (Barres)"):
    """
    Affiche les métriques de toutes les solutions de l'archive sous forme de graphiques à barres.
    Triées par LBI (du pire au meilleur).
    """
    if not metrics_data.get('LBI'):
        print("Aucune métrique à afficher pour les barres.")
        return None
    
    # Trier par LBI décroissant (pire → meilleur, car LBI élevé = moins bon)
    lbi_values = np.array(metrics_data['LBI'])
    sorted_indices = np.argsort(lbi_values)[::-1]
    
    # Réorganiser toutes les métriques
    metrics_data_sorted = {}
    for key in metrics_data.keys():
        metrics_data_sorted[key] = [metrics_data[key][i] for i in sorted_indices]
    
    solution_indices = np.arange(1, n_solutions + 1)
    
    fig = plt.figure(figsize=(18, 12)) # Taille ajustée pour 5 barres
    
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
        },
        'AvgCost': { # NEW METRIC
            'color': 'darkgoldenrod', 'edgecolor': 'saddlebrown',
            'title': 'Average Cost per Video (AvgCost)', 'ylabel': 'Avg Cost',
            'better': 'Plus bas = meilleur', 'best_func': np.argmin,
            'best_label': 'Min Coût Moy.'
        }
    }
    
    for idx, (metric_name, config) in enumerate(metrics_config.items(), 1):
        # Utilise plt.subplot(2, 3, idx) pour une grille 2x3 qui peut contenir 5 plots
        if metric_name in metrics_data_sorted:
            ax = plt.subplot(2, 3, idx) 
            _create_metric_subplot(ax, solution_indices, metrics_data_sorted[metric_name], 
                                  metric_name, config)
    
    plt.suptitle(title + "\n(Triées par LBI: Pire → Meilleur)", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig

def plot_archive_text_summary(metrics_data, archive_solutions, title="Analyse des Métriques - Résumé"):
    """
    Affiche les recommandations et les statistiques descriptives.
    """
    n_solutions = len(archive_solutions)
    if n_solutions == 0:
        print("Aucune solution dans l'archive pour le résumé.")
        return None
        
    fig = plt.figure(figsize=(15, 7)) # Nouvelle figure pour les textes
    
    # Recommandations (subplot 1)
    ax1 = plt.subplot(1, 2, 1) # 1x2 grid
    ax1.axis('off')
    
    scores = compute_composite_scores(metrics_data, n_solutions)
    top3_indices = np.argsort(scores)[-3:][::-1]
    
    recommendation_text = "RECOMMANDATIONS\n" + "="*50 + "\n"
    recommendation_text += "TOP 3 SOLUTIONS (score composite):\n"
    
    for rank, idx in enumerate(top3_indices, 1):
        sol_num = idx + 1
        recommendation_text += f"#{rank} - Solution {sol_num} (score: {scores[idx]:.3f})\n"
        recommendation_text += f"   LBI: {metrics_data['LBI'][idx]:.3f} \n"
        recommendation_text += f"   FUR: {metrics_data['FUR'][idx]:.3f} \n"
        recommendation_text += f"   EE: {metrics_data['EE'][idx]:.1f} \n"
        recommendation_text += f"   Latency: {metrics_data['AvgLatency'][idx]:.3f} \n"
        if 'AvgCost' in metrics_data:
            recommendation_text += f"   AvgCost: {metrics_data['AvgCost'][idx]:.3f} \n" # Affichage AvgCost
        recommendation_text += f"\n"
    
    ax1.text(0.05, 0.95, recommendation_text, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', 
                      alpha=0.8, edgecolor='blue', linewidth=2))
    
    # Statistiques (subplot 2)
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')
    stats_text = "STATISTIQUES DESCRIPTIVES\n" + "="*40 + "\n\n"
    
    metrics_to_show = ['LBI', 'FUR', 'EE', 'AvgLatency']
    if 'AvgCost' in metrics_data:
        metrics_to_show.append('AvgCost')
        
    for metric_name in metrics_to_show:
        values = np.array(metrics_data[metric_name])
        stats_text += f"{metric_name}:\n"
        stats_text += f"  Moyenne: {np.mean(values):.4f}\n"
        stats_text += f"  Médiane: {np.median(values):.4f}\n"
        stats_text += f"  Écart-type: {np.std(values):.4f}\n"
        stats_text += f"  Min: {np.min(values):.4f}\n"
        stats_text += f"  Max: {np.max(values):.4f}\n\n"
    
    ax2.text(0.1, 0.95, stats_text, transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', 
                      alpha=0.8, edgecolor='orange', linewidth=2))
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig