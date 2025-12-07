"""
Module de métriques pour l'optimisation multi-objectifs.
Contient les calculs de métriques de performance et de visualisation.
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# MÉTRIQUES DE BASE ET HYPERVOLUME
# =============================================================================

def extract_objectives(solutions):
    """Transforme une liste de Solution en matrice (N,3): [makespan, cost, energy]."""
    return np.array([[s.makespan, s.cost, s.energy] for s in solutions], dtype=float)


def compute_reference_point(objs, factor=1.1):
    """Définit un point de référence un peu pire que le max des objectifs."""
    return objs.max(axis=0) * factor


def hypervolume_3d(objs, ref_point):
    """
    Calcule l'hypervolume 3D (minimisation) approximatif.
    Plus HV est large, plus l'ensemble de solutions est proche du coin idéal
    et bien étendu dans l'espace.
    
    Args:
        objs: array (N,3) - [makespan, cost, energy]
        ref_point: array-like (3,) - point de référence
    
    Returns:
        float: valeur de l'hypervolume
    """
    idx = np.argsort(objs[:, 0])
    sorted_objs = objs[idx]
    
    hv = 0.0
    prev_m = ref_point[0]
    
    for i in range(sorted_objs.shape[0]):
        m, c, e = sorted_objs[i]
        dm = prev_m - m
        if dm <= 0:
            prev_m = m
            continue
        
        sub = sorted_objs[i:, 1:]
        best_c = sub[:, 0].min()
        best_e = sub[:, 1].min()
        dc = ref_point[1] - best_c
        de = ref_point[2] - best_e
        
        if dc > 0 and de > 0:
            hv += dm * dc * de
        prev_m = m
    
    return hv


def init_hv_tracking(initial_archive):
    """Initialise le suivi d'hypervolume. Retourne (hv_history, ref_point)."""
    objs = extract_objectives(initial_archive)
    ref_point = compute_reference_point(objs, factor=1.1)
    hv0 = hypervolume_3d(objs, ref_point)
    return [hv0], ref_point


def update_hv_tracking(archive_solutions, hv_history, ref_point):
    """Met à jour l'historique d'hypervolume à une itération donnée."""
    objs = extract_objectives(archive_solutions)
    hv = hypervolume_3d(objs, ref_point)
    hv_history.append(hv)
    return hv_history


# =============================================================================
# MÉTRIQUES DE QUALITÉ DES SOLUTIONS
# =============================================================================

def load_balancing_index(solution, donnees):
    """
    Load Balancing Index: LBI = std(load_j) / mean(load_j).
    Plus LBI est faible, meilleur est l'équilibrage de charge.
    """
    loads = np.zeros(donnees.p, dtype=float)
    for i in range(donnees.n):
        vm = solution.assignment[i]
        loads[vm] += donnees.U_ij[i, vm]
    
    mean_load = loads.mean()
    if mean_load == 0:
        return 0.0
    
    return loads.std() / mean_load


def fog_utilization_ratio(solution, donnees):
    """
    Fog Utilization Ratio: FUR = (# vidéos sur VMs Fog) / n.
    Plus FUR est élevé, meilleure est l'utilisation du fog.
    """
    count_fog = sum(1 for i in range(donnees.n) 
                    if donnees.is_fog_j[solution.assignment[i]])
    return count_fog / donnees.n


def energy_efficiency(solution):
    """
    Energy Efficiency: EE = energy / makespan.
    Énergie consommée par unité de temps.
    """
    if solution.makespan == 0:
        return np.inf
    return solution.energy / solution.makespan


def average_latency(solution, donnees):
    """
    Latence moyenne: (1/n) * sum_i U_ij(i, vm(i)).
    Plus la latence est faible, meilleure est la performance.
    """
    total_latency = sum(donnees.U_ij[i, solution.assignment[i]] 
                        for i in range(donnees.n))
    return total_latency / donnees.n


# =============================================================================
# MÉTRIQUES DE DIVERSITÉ ET QUALITÉ PARETO
# =============================================================================

def diversity_spread(objs):
    """
    Mesure de diversité: longueur moyenne des segments entre points voisins.
    Plus la valeur est élevée, plus la Pareto front est bien étalée.
    """
    if len(objs) < 2:
        return 0.0
    
    idx = np.argsort(objs[:, 0])
    sorted_objs = objs[idx]
    diffs = np.linalg.norm(sorted_objs[1:] - sorted_objs[:-1], axis=1)
    return diffs.mean()


def pareto_size(objs):
    """
    Taille de la frontière de Pareto.
    Plus la taille est grande, plus on offre de compromis au décideur.
    """
    return len(objs)


def spacing_metric(objs):
    """
    Spacing metric pour mesurer la régularité des solutions.
    Plus SP est faible, plus les solutions sont régulièrement espacées.
    """
    if len(objs) < 2:
        return 0.0
    
    distances = []
    for i in range(len(objs)):
        dists = np.linalg.norm(objs[i] - objs, axis=1)
        dists = dists[dists != 0]
        if len(dists) > 0:
            distances.append(dists.min())
    
    if not distances:
        return 0.0
    
    distances = np.array(distances)
    return distances.std()


# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_hv_convergence(hv_history, title="Convergence de l'hypervolume"):
    """Trace la courbe de convergence de l'hypervolume au fil des itérations."""
    if not hv_history:
        print("hv_history est vide, rien à tracer.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(hv_history)), hv_history, marker="o")
    plt.xlabel("Itération", fontsize=12)
    plt.ylabel("Hypervolume", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_pareto_2d(objs, x_idx=0, y_idx=1, x_label="Makespan", y_label="Cost"):
    """
    Visualisation 2D de la Pareto front.
    La couleur représente l'énergie.
    
    Args:
        objs: array (N,3) = [makespan, cost, energy]
        x_idx, y_idx: indices des objectifs à mettre en x et y (0,1,2)
    """
    if objs is None or len(objs) == 0:
        print("objs est vide, rien à tracer.")
        return

    x = objs[:, x_idx]
    y = objs[:, y_idx]
    energy = objs[:, 2]

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(x, y, c=energy, s=70, edgecolor="black", alpha=0.85)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title("Distribution des solutions Pareto (2D)", fontsize=14)
    cbar = plt.colorbar(sc)
    cbar.set_label("Energy", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# =============================================================================
# CALCUL ET VISUALISATION DES MÉTRIQUES POUR L'ARCHIVE
# =============================================================================

def compute_metrics_all_solutions(archive_solutions, donnees):
    """
    Calcule et affiche les métriques pour toutes les solutions de l'archive.
    
    Args:
        archive_solutions: liste des solutions de l'archive
        donnees: données du problème
        
    Returns:
        dict: dictionnaire contenant les listes de métriques pour toutes les solutions
    """
    print("\n" + "="*70)
    print("MÉTRIQUES FINALES - TOUTES LES SOLUTIONS DE L'ARCHIVE")
    print("="*70)
    
    metrics_data = {
        'LBI': [],
        'FUR': [],
        'EE': [],
        'AvgLatency': []
    }
    
    if archive_solutions:
        for idx, solution in enumerate(archive_solutions, 1):
            print(f"\n--- Solution {idx} ---")
            lbi = load_balancing_index(solution, donnees)
            fur = fog_utilization_ratio(solution, donnees)
            ee = energy_efficiency(solution)
            avg_lat = average_latency(solution, donnees)
            
            print(f"[Metrics SOLUTION] LBI={lbi:.4f}, FUR={fur:.4f}, "
                  f"EE={ee:.4f}, AvgLatency={avg_lat:.4f}")
            
            metrics_data['LBI'].append(lbi)
            metrics_data['FUR'].append(fur)
            metrics_data['EE'].append(ee)
            metrics_data['AvgLatency'].append(avg_lat)
    else:
        print("Aucune solution dans l'archive")
    
    return metrics_data


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


def _compute_composite_scores(metrics_data, n_solutions):
    """Calcule les scores composites normalisés pour les solutions."""
    scores = []
    
    for i in range(n_solutions):
        lbi_norm = 1 - (metrics_data['LBI'][i] - min(metrics_data['LBI'])) / \
                   (max(metrics_data['LBI']) - min(metrics_data['LBI']) + 1e-4)
        fur_norm = (metrics_data['FUR'][i] - min(metrics_data['FUR'])) / \
                   (max(metrics_data['FUR']) - min(metrics_data['FUR']) + 1e-4)
        ee_norm = (metrics_data['EE'][i] - min(metrics_data['EE'])) / \
                  (max(metrics_data['EE']) - min(metrics_data['EE']) + 1e-4)
        lat_norm = 1 - (metrics_data['AvgLatency'][i] - min(metrics_data['AvgLatency'])) / \
                   (max(metrics_data['AvgLatency']) - min(metrics_data['AvgLatency']) + 1e-4)
        
        score = 0.25 * (lbi_norm + fur_norm + ee_norm + lat_norm)
        scores.append(score)
    
    return scores


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
    
    recommendation_text = "RECOMMANDATIONS\n" + "="*50 + "\n\n"
    recommendation_text += "TOP 3 SOLUTIONS (score composite):\n\n"
    
    for rank, idx in enumerate(top3_indices, 1):
        sol_num = idx + 1
        recommendation_text += f"#{rank} - Solution {sol_num} (score: {scores[idx]:.3f})\n"
        recommendation_text += f"   LBI: {metrics_data['LBI'][idx]:.3f} "
        recommendation_text += f"{'OK' if metrics_data['LBI'][idx] < np.mean(metrics_data['LBI']) else 'X'}\n"
        recommendation_text += f"   FUR: {metrics_data['FUR'][idx]:.3f} "
        recommendation_text += f"{'OK' if metrics_data['FUR'][idx] > np.mean(metrics_data['FUR']) else 'X'}\n"
        recommendation_text += f"   EE: {metrics_data['EE'][idx]:.1f} "
        recommendation_text += f"{'OK' if metrics_data['EE'][idx] > np.mean(metrics_data['EE']) else 'X'}\n"
        recommendation_text += f"   Latency: {metrics_data['AvgLatency'][idx]:.3f} "
        recommendation_text += f"{'OK' if metrics_data['AvgLatency'][idx] < np.mean(metrics_data['AvgLatency']) else 'X'}\n\n"
    
    recommendation_text += "\nINTERPRETATION:\n"
    recommendation_text += "OK = Au-dessus de la moyenne (bon)\n"
    recommendation_text += "X  = En-dessous de la moyenne\n"
    
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
    plt.show()


# =============================================================================
# FONCTIONS D'AFFICHAGE COMBINÉES
# =============================================================================

def compute_and_display_metrics(target, metric_type, donnees=None, hv_history=None, ref_point=None):
    """
    Calcule et affiche les métriques selon le type demandé.
    
    Args:
        target: Archive (list) ou solution individuelle
        metric_type: 'archive' ou 'solution'
        donnees: Données du problème (requis pour metric_type='solution')
        hv_history: Historique hypervolume (pour metric_type='archive')
        ref_point: Point de référence (pour metric_type='archive')
        
    Returns:
        tuple: (hv_history, ref_point) si metric_type='archive', sinon None
    """
    if metric_type == 'archive' and target:
        objs_arch = extract_objectives(target)
        
        if hv_history is None:
            hv_history, ref_point = init_hv_tracking(target)
        else:
            hv_history = update_hv_tracking(target, hv_history, ref_point)

        div = diversity_spread(objs_arch)
        sp = spacing_metric(objs_arch)
        psize = pareto_size(objs_arch)
        
        print(f"\n[Metrics ARCHIVE] HV={hv_history[-1]:.4f}, "
              f"Diversity={div:.4f}, Spacing={sp:.4f}, Pareto size={psize}")
        
        return hv_history, ref_point
    
    elif metric_type == 'solution' and target is not None and donnees is not None:
        lbi = load_balancing_index(target, donnees)
        fur = fog_utilization_ratio(target, donnees)
        ee = energy_efficiency(target)
        avg_lat = average_latency(target, donnees)
        
        print(f"\n[Metrics SOLUTION] LBI={lbi:.4f}, FUR={fur:.4f}, "
              f"EE={ee:.4f}, AvgLatency={avg_lat:.4f}")
        
        return None
    
    return hv_history, ref_point


def compute_and_display_archive_metrics(archive_solutions, hv_history, ref_point):
    """Calcule et affiche les métriques de l'archive."""
    return compute_and_display_metrics(archive_solutions, 'archive', 
                                      hv_history=hv_history, ref_point=ref_point)


def compute_and_display_alpha_metrics(alpha, donnees):
    """Calcule et affiche les métriques du leader alpha."""
    compute_and_display_metrics(alpha, 'solution', donnees=donnees)


def compute_and_display_archive_solutions_metrics(archive_solutions, donnees):
    """
    Calcule et affiche les métriques pour toutes les solutions de l'archive.
    
    Args:
        archive_solutions: Liste des solutions de l'archive
        donnees: Données du problème
    """
    print("\n" + "="*70)
    print("MÉTRIQUES FINALES - TOUTES LES SOLUTIONS DE L'ARCHIVE")
    print("="*70)
    
    if archive_solutions:
        for idx, solution in enumerate(archive_solutions, 1):
            print(f"\n--- Solution {idx} ---")
            compute_and_display_metrics(solution, 'solution', donnees=donnees)
    else:
        print("Aucune solution dans l'archive")

