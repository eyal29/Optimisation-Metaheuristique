import numpy as np
import matplotlib.pyplot as plt
# import time

def extract_objectives(solutions):
    """
    Transforme une liste de Solution en matrice (N,3):
    [makespan, cost, energy].
    """
    objs = np.array([
        [s.makespan, s.cost, s.energy]
        for s in solutions
    ], dtype=float)
    return objs

def compute_reference_point(objs, factor=1.1):
    """
    Définit un point de référence un peu pire que le max des objectifs.
    """
    worst = objs.max(axis=0)
    return worst * factor


def hypervolume_3d(objs, ref_point):
    """
    Mesure la qualité globale de la frontière :
    plus HV est large (pour un même point de référence), plus l'ensemble de solutions est :
    proche du “coin idéal” (low makespan, low cost, low energy),
    et bien étendu dans l’espace.

    Hypervolume 3D (minimisation) approximatif.
    objs : array (N,3)
    ref_point : array-like (3,)
    """
    # On travaille en minimisation => on considère le volume entre chaque point et ref
    # Tri par le premier objectif (makespan) croissant
    idx = np.argsort(objs[:, 0])
    sorted_objs = objs[idx]

    hv = 0.0
    # On va considérer des "tranches" successives
    prev_m = ref_point[0]
    for i in range(sorted_objs.shape[0]):
        m, c, e = sorted_objs[i]
        dm = prev_m - m
        if dm <= 0:
            prev_m = m
            continue
        # pour cette tranche de makespan, on prend le minimum cost/energy dans les points restants
        sub = sorted_objs[i:, 1:]  # (cost, energy)
        # simple approximation: prendre le min sur cost et energy
        best_c = sub[:, 0].min()
        best_e = sub[:, 1].min()
        dc = ref_point[1] - best_c
        de = ref_point[2] - best_e
        if dc > 0 and de > 0:
            hv += dm * dc * de
        prev_m = m
    return hv

# ---------------------------------------------------------
# 2. Taux de convergence du HV
# ---------------------------------------------------------

def init_hv_tracking(initial_archive):
    """
    Initialise le suivi d'hypervolume.
    Retourne (hv_history, ref_point).
    """
    objs = extract_objectives(initial_archive)
    ref_point = compute_reference_point(objs, factor=1.1)
    hv0 = hypervolume_3d(objs, ref_point)
    return [hv0], ref_point


def update_hv_tracking(archive_solutions, hv_history, ref_point):
    """
    Met à jour l'historique d'hypervolume à une itération donnée.
    """
    objs = extract_objectives(archive_solutions)
    hv = hypervolume_3d(objs, ref_point)
    hv_history.append(hv)
    return hv_history

# ---------------------------------------------------------
# 3. Load Balancing Index (LBI)
# ---------------------------------------------------------

def load_balancing_index(solution, donnees):
    """
    LBI = std(load_j) / mean(load_j)
    - solution : Solution (avec assignment, makespan, etc.)
    - donnees : Donnees (contenant U_ij, n, p)
    """
    n, p = donnees.n, donnees.p
    U_ij = donnees.U_ij

    loads = np.zeros(p, dtype=float)
    for i in range(n):
        vm = solution.assignment[i]
        loads[vm] += U_ij[i, vm]

    mean_load = loads.mean()
    if mean_load == 0:
        return 0.0  # évite division par zéro

    lbi = np.sqrt(((loads - mean_load) ** 2).mean()) / mean_load
    return lbi

# ---------------------------------------------------------
# 4. Fog Utilization Ratio (FUR)
# ---------------------------------------------------------
def fog_utilization_ratio(solution, donnees):
    """
    FUR = (# vidéos exécutées sur des VMs Fog) / n
    """
    # On suppose maintenant que Donnees définit TOUJOURS is_fog_j
    is_fog = donnees.is_fog_j   # shape (p,), bool
    n = donnees.n

    count_fog = 0
    for i in range(n):
        vm = solution.assignment[i]
        if is_fog[vm]:
            count_fog += 1

    return count_fog / n



# ---------------------------------------------------------
# 5. Energy Efficiency (EE)
# ---------------------------------------------------------

def energy_efficiency(solution):
    """
    EE = total_energy / makespan
    -> énergie consommée par unité de temps.

    solution.energy et solution.makespan doivent déjà être évalués.
    """
    if solution.makespan == 0:
        return np.inf
    return solution.energy / solution.makespan


# ---------------------------------------------------------
# 6. Average Latency
# ---------------------------------------------------------

def average_latency(solution, donnees):
    """
    Latence moyenne = (1/n) * sum_i U_ij(i, vm(i))

    - n : nombre de vidéos
    - U_ij : matrice temps total (transfert + traitement + sortie)
    """
    n = donnees.n
    U_ij = donnees.U_ij

    total_latency = 0.0
    for i in range(n):
        vm = solution.assignment[i]
        total_latency += U_ij[i, vm]

    return total_latency / n



def diversity_spread(objs):
    """
    Si la valeur est faible → les solutions sont concentrées dans une zone → tu “rates” certains compromis intéressants.
    Si la valeur est plus élevée → la Pareto front est bien “étalée” entre par ex.
    solutions très rapides mais chères/ économiques mais plus lentes/ solutions intermédiaires

    Mesure simple de diversité : longueur moyenne des segments
    entre points voisins (après tri par makespan).
    """
    if len(objs) < 2:
        return 0.0
    idx = np.argsort(objs[:, 0])
    sorted_objs = objs[idx]
    diffs = np.linalg.norm(
        sorted_objs[1:] - sorted_objs[:-1],
        axis=1
    )
    return diffs.mean()


def pareto_size(objs):
    """
    Plus il est grand, plus on offre de compromis au décideur :
    solutions “low latency / high cost”,
    solutions “low cost / higher makespan”,
    Si size est trop petit, l'algo explore mal.
    """
    return len(objs)

def spacing_metric(objs):
    """
    Spacing metric SP pour mesurer la régularité des solutions.
    Plus SP est faible, plus les solutions sont régulièrement espacées.
    """
    if len(objs) < 2:
        return 0.0

    distances = []
    for i in range(len(objs)):
        dists = np.linalg.norm(objs[i] - objs, axis=1)
        dists = dists[dists != 0]  # enlever soi-même
        distances.append(dists.min())

    distances = np.array(distances)
    return np.sqrt(((distances - distances.mean()) ** 2).mean())





# ---------------------------------------------------------
# 7. PLots des métriques
# ---------------------------------------------------------

def plot_hv_convergence(hv_history, title="Convergence de l'hypervolume"):
    """
    Trace la courbe de convergence de l'hypervolume au fil des itérations.
    
    hv_history : liste des valeurs d'hypervolume, hv_history[t] = HV à l'itération t.
    """
    if len(hv_history) == 0:
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


def plot_metrics_bar(metrics_dict, title="Métriques de performance (Fog-Cloud)"):
    """
    Trace un barplot des métriques scalaires (LBI, FUR, EE, Latence, etc.).
    
    metrics_dict : dict {nom_metric: valeur}
        Exemple :
        {
            "LBI": 0.12,
            "FUR": 0.68,
            "EE":  150.3,
            "Avg Latency": 2.35
        }
    """
    if not metrics_dict:
        print("metrics_dict est vide, rien à tracer.")
        return

    names = list(metrics_dict.keys())
    values = [metrics_dict[k] for k in names]

    plt.figure(figsize=(8, 5))
    x = np.arange(len(names))
    plt.bar(x, values)
    plt.xticks(x, names, rotation=20)
    plt.ylabel("Valeur", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_pareto_2d(objs, x_idx=0, y_idx=1,
                   x_label="Makespan", y_label="Cost"):
    """
    Visualisation 2D simple de la Pareto front :
    - par défaut : Makespan (x) vs Cost (y)
    - la couleur représente l'énergie.
    
    objs : array (N,3) = [makespan, cost, energy]
    x_idx, y_idx : indices des objectifs à mettre en x et y (0,1,2)
    """
    if objs is None or len(objs) == 0:
        print("objs est vide, rien à tracer.")
        return

    x = objs[:, x_idx]
    y = objs[:, y_idx]
    energy = objs[:, 2]  # par convention : 2 = energy

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


# temps d'execution global
# start = time.time()
# # ... exécution de l'algorithme
# end = time.time()
# elapsed = end - start
# print(f"Temps total d'exécution : {elapsed:.2f} s")
# print(f"Nombre total d'évaluations : {nb_eval}")

def plot_metrics_subplots(metrics_dict, title="Métriques Fog-Cloud (meilleure solution archive)"):
    """
    Affiche les métriques Fog-Cloud sur 4 sous-graphiques avec :
    - échelles indépendantes
    - barres centrées
    - valeurs numériques affichées
    - tailles visuelles harmonisées
    """
    if not metrics_dict:
        print("metrics_dict est vide, rien à tracer.")
        return

    names = list(metrics_dict.keys())
    values = [metrics_dict[k] for k in names]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for i, (name, value) in enumerate(zip(names, values)):
        ax = axes[i]

        # Barre
        ax.bar([0], [value], width=0.4)

        # Titre du sous-graphe
        ax.set_title(name, fontsize=12)

        # Affichage de la valeur numérique au-dessus de la barre
        ax.text(
            0, value, f"{value:.4g}",
            ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )

        # Pas de ticks sur x
        ax.set_xticks([])

        # Ajout d'une grille légère
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        # Marges pour éviter que la barre touche le haut
        ax.set_ylim(0, value * 1.25 if value > 0 else 1)

    # Masque les cases vides si < 4 métriques
    for j in range(len(names), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()