import numpy as np
import matplotlib.pyplot as plt


# MÉTRIQUES DE BASE ET HYPERVOLUME
def extract_objectives(solutions):
    return np.array([[s.makespan, s.cost, s.energy] for s in solutions], dtype=float)


def compute_reference_point(objs, factor=1.1):
    return objs.max(axis=0) * factor


def hypervolume_3d(objs, ref_point):
    """
    Calcule l'hypervolume 3D (minimisation) approximatif.
    Plus HV est large, plus l'ensemble de solutions est proche du coin idéal
    et bien étendu dans l'espace.
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
    objs = extract_objectives(initial_archive)
    ref_point = compute_reference_point(objs, factor=1.1)
    hv0 = hypervolume_3d(objs, ref_point)
    return [hv0], ref_point


def update_hv_tracking(archive_solutions, hv_history, ref_point):
    objs = extract_objectives(archive_solutions)
    hv = hypervolume_3d(objs, ref_point)
    hv_history.append(hv)
    return hv_history


# MÉTRIQUES DE QUALITÉ DES SOLUTIONS
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



# MÉTRIQUES DE DIVERSITÉ ET QUALITÉ PARETO
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


# CALCUL ET VISUALISATION DES MÉTRIQUES POUR L'ARCHIVE
def compute_metrics_all_solutions(archive_solutions, donnees):
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


def compute_composite_scores(metrics_data, n_solutions):
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


