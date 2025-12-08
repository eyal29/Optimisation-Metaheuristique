import time
import random
import numpy as np
import matplotlib.pyplot as plt 
from utils import load_config, load_data
from solutions_definiton import Donnees
from pareto import ParetoArchive, select_leaders
from algo_utils import generate_valid_solution, compute_a, gwo_update_population
from metrics import *

def run_gwo_single(
    pop_size,
    max_iter,
    archive_max,
    a_max=None,
    a_min=None,
    seed=None,
    patience=None,
    min_delta=1e-6,
):
    """
    Lance UN run de GWO-NSGA-II et renvoie un dict avec :
    - hv_history, final_hv, final_size, final_div, final_sp, exec_time
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    config = load_config("config.yaml")

    if a_max is None:
        a_max = config["gwo"]["a_max"]
    if a_min is None:
        a_min = config["gwo"]["a_min"]

    videos, vms = load_data(config["paths"]["videos"], config["paths"]["vms"])
    donnees = Donnees(videos, vms)
    archive = ParetoArchive(max_size=archive_max)
    hv_history, ref_point = None, None

    population = [generate_valid_solution(donnees) for _ in range(pop_size)]
    start_time = time.time()

    # early stopping sur l'hypervolume
    best_hv = -float("inf")
    no_improve = 0

    for t in range(1, max_iter + 1):
        a = compute_a(t, max_iter, a_max=a_max, a_min=a_min)

        # évaluation + mise à jour de l'archive
        for sol in population:
            sol.evaluate()
            archive.add(sol)

        archive_solutions = archive.get_solutions()
        # hypervolume + early stopping
        if archive_solutions:
            if hv_history is None:
                hv_history, ref_point = init_hv_tracking(archive_solutions)
            else:
                hv_history = update_hv_tracking(archive_solutions, hv_history, ref_point)

            current_hv = hv_history[-1]

            if current_hv > best_hv + min_delta:
                best_hv = current_hv
                no_improve = 0
            else:
                no_improve += 1

            if patience is not None and no_improve >= patience:
                print(
                    f"[Early stopping] arrêt à l’itération {t} "
                    f"(aucune amélioration HV depuis {patience} itérations)"
                )
                break

        # leaders + mise à jour GWO
        alpha, beta, delta = select_leaders(population)
        population = gwo_update_population(population, alpha, beta, delta, donnees, a)

    exec_time = time.time() - start_time

    archive_finale = archive.get_solutions()
    if archive_finale:
        objs_final = extract_objectives(archive_finale)
        final_hv = hv_history[-1] if hv_history is not None else None
        final_size = pareto_size(objs_final)
        final_div = diversity_spread(objs_final)
        final_sp = spacing_metric(objs_final)
    else:
        final_hv = final_size = final_div = final_sp = None

    return {
        "hv_history": hv_history,
        "final_hv": final_hv,
        "final_size": final_size,
        "final_div": final_div,
        "final_sp": final_sp,
        "exec_time": exec_time,
    }


def run_experiments():
    # Ajustement des configs
    param_configs = [
        (
            "C5: pop=200, iter=150, arch=200",
            dict(pop_size=200, max_iter=150, archive_max=200),
        ),
        (
            "C6: pop=200, iter=250, arch=200",
            dict(pop_size=200, max_iter=250, archive_max=200),
        ),
    ]
    n_runs = 4
    all_results = {}

    # -------- Lancer tous les runs --------
    for name, params in param_configs:
        print(f"\n=== {name} ===")
        hv_histories = []
        final_hv = []
        final_size = []
        final_div = []
        final_sp = []
        exec_times = []

        for r in range(n_runs):
            print(f"  -> Run {r + 1}/{n_runs}")
            res = run_gwo_single(seed=r, patience=30, **params)

            hv_histories.append(res["hv_history"])
            final_hv.append(res["final_hv"])
            final_size.append(res["final_size"])
            final_div.append(res["final_div"])
            final_sp.append(res["final_sp"])
            exec_times.append(res["exec_time"])

        all_results[name] = {
            "params": params,
            "hv_histories": hv_histories,
            "final_hv": np.array(final_hv, dtype=float),
            "final_size": np.array(final_size, dtype=float),
            "final_div": np.array(final_div, dtype=float),
            "final_sp": np.array(final_sp, dtype=float),
            "exec_times": np.array(exec_times, dtype=float),
        }

    print("\n\n================= RÉSUMÉ STATISTIQUE =================")
    print("Config | HV moyen ± éc.-type | Taille Pareto | Diversité | Spacing | Temps moyen (s)")
    print("--------------------------------------------------------------------------")
    for name, info in all_results.items():
        hv = info["final_hv"]
        sz = info["final_size"]
        dv = info["final_div"]
        sp = info["final_sp"]
        tm = info["exec_times"]

        print(
            f"{name:30s} | "
            f"{np.nanmean(hv):.3e} ± {np.nanstd(hv):.2e} | "
            f"{np.nanmean(sz):6.1f} | "
            f"{np.nanmean(dv):7.3f} | "
            f"{np.nanmean(sp):7.3f} | "
            f"{np.nanmean(tm):7.2f}"
        )

    # ========= FIGURES MATPLOTLIB CLASSIQUES (une fenêtre par figure) =========

    labels = list(all_results.keys())
    x = np.arange(len(labels))

    # 1) Convergence de l'hypervolume
    fig = plt.figure(figsize=(10, 6))
    for name, info in all_results.items():
        hv_histories = info["hv_histories"]

        max_len = max(len(hv) for hv in hv_histories)
        hv_mat = np.full((len(hv_histories), max_len), np.nan)
        for i, hv in enumerate(hv_histories):
            hv_mat[i, :len(hv)] = hv

        mean_hv = np.nanmean(hv_mat, axis=0)
        std_hv = np.nanstd(hv_mat, axis=0)
        iters = np.arange(1, max_len + 1)

        plt.plot(iters, mean_hv, marker="o", label=name)
        plt.fill_between(iters, mean_hv - std_hv, mean_hv + std_hv, alpha=0.15)

    plt.xlabel("Itération")
    plt.ylabel("Hypervolume")
    plt.title("Convergence de l'hypervolume (moyenne ± écart-type)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    # 2) Hypervolume final (barres + barres d'erreur)
    hv_means = [np.nanmean(info["final_hv"]) for info in all_results.values()]
    hv_stds = [np.nanstd(info["final_hv"]) for info in all_results.values()]

    fig = plt.figure(figsize=(10, 5))
    plt.bar(x, hv_means, yerr=hv_stds, capsize=5)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Hypervolume final moyen")
    plt.title("Comparaison des hypervolumes finaux")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    # 3) Taille Pareto, Diversité, Spacing
    size_means = [np.nanmean(info["final_size"]) for info in all_results.values()]
    div_means = [np.nanmean(info["final_div"]) for info in all_results.values()]
    sp_means = [np.nanmean(info["final_sp"]) for info in all_results.values()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

    axes[0].bar(x, size_means)
    axes[0].set_title("Taille Pareto (moyenne)")
    axes[0].set_ylabel("Nb solutions")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    axes[1].bar(x, div_means)
    axes[1].set_title("Diversité (spread)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    axes[2].bar(x, sp_means)
    axes[2].set_title("Spacing")
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)

    fig.suptitle("Structure du front Pareto pour chaque configuration", y=1.05)
    plt.tight_layout()

    # 4) Temps d'exécution moyen
    time_means = [np.nanmean(info["exec_times"]) for info in all_results.values()]

    fig = plt.figure(figsize=(10, 5))
    plt.bar(x, time_means)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Temps moyen (s)")
    plt.title("Coût de calcul par configuration")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    # Affiche toutes les figures dans des fenêtres séparées
    plt.show()


if __name__ == "__main__":
    run_experiments()
