"""
Module pour la visualisation des métriques de l'archive.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_archive_metrics_visualization(metrics_data, archive_solutions):
    """
    Affiche les métriques de toutes les solutions de l'archive sous forme de graphiques.
    
    Args:
        metrics_data: Dict contenant 'LBI', 'FUR', 'EE', 'AvgLatency'
        archive_solutions: Liste des solutions de l'archive
    """
    if not metrics_data['LBI']:
        print("Aucune métrique à afficher.")
        return
    
    n_solutions = len(metrics_data['LBI'])
    solution_indices = np.arange(1, n_solutions + 1)
    
    # Créer une figure avec 5 sous-graphiques (2x3 grid)
    fig = plt.figure(figsize=(15, 8))
    
    # 1. Graphique en barres pour LBI
    ax1 = plt.subplot(2, 3, 1)
    bars1 = ax1.bar(solution_indices, metrics_data['LBI'], color='steelblue', alpha=0.7, edgecolor='navy')
    ax1.set_xlabel('Solution', fontweight='bold')
    ax1.set_ylabel('LBI', fontweight='bold')
    ax1.set_title('Load Balancing Index (LBI)', fontweight='bold', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=np.mean(metrics_data['LBI']), color='red', linestyle='--', linewidth=2, label=f'Moyenne: {np.mean(metrics_data["LBI"]):.3f}')
    ax1.legend()
    
    # 2. Graphique en barres pour FUR
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.bar(solution_indices, metrics_data['FUR'], color='seagreen', alpha=0.7, edgecolor='darkgreen')
    ax2.set_xlabel('Solution', fontweight='bold')
    ax2.set_ylabel('FUR', fontweight='bold')
    ax2.set_title('Fog Utilization Ratio (FUR)', fontweight='bold', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=np.mean(metrics_data['FUR']), color='red', linestyle='--', linewidth=2, label=f'Moyenne: {np.mean(metrics_data["FUR"]):.3f}')
    ax2.legend()
    
    # 3. Graphique en barres pour EE
    ax3 = plt.subplot(2, 3, 3)
    bars3 = ax3.bar(solution_indices, metrics_data['EE'], color='coral', alpha=0.7, edgecolor='darkred')
    ax3.set_xlabel('Solution', fontweight='bold')
    ax3.set_ylabel('EE', fontweight='bold')
    ax3.set_title('Energy Efficiency (EE)', fontweight='bold', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=np.mean(metrics_data['EE']), color='blue', linestyle='--', linewidth=2, label=f'Moyenne: {np.mean(metrics_data["EE"]):.3f}')
    ax3.legend()
    
    # 4. Graphique en barres pour AvgLatency
    ax4 = plt.subplot(2, 3, 4)
    bars4 = ax4.bar(solution_indices, metrics_data['AvgLatency'], color='mediumpurple', alpha=0.7, edgecolor='indigo')
    ax4.set_xlabel('Solution', fontweight='bold')
    ax4.set_ylabel('Latence Moyenne', fontweight='bold')
    ax4.set_title('Average Latency', fontweight='bold', fontsize=12)
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=np.mean(metrics_data['AvgLatency']), color='red', linestyle='--', linewidth=2, label=f'Moyenne: {np.mean(metrics_data["AvgLatency"]):.3f}')
    ax4.legend()
    
    # 5. Statistiques textuelles
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    stats_text = "STATISTIQUES DESCRIPTIVES\n" + "="*40 + "\n\n"
    
    for metric_name in ['LBI', 'FUR', 'EE', 'AvgLatency']:
        values = np.array(metrics_data[metric_name])
        stats_text += f"{metric_name}:\n"
        stats_text += f"  Moyenne: {np.mean(values):.4f}\n"
        stats_text += f"  Médiane: {np.median(values):.4f}\n"
        stats_text += f"  Écart-type: {np.std(values):.4f}\n"
        stats_text += f"  Min: {np.min(values):.4f}\n"
        stats_text += f"  Max: {np.max(values):.4f}\n\n"
    
    ax5.text(0.1, 0.95, stats_text, transform=ax5.transAxes, 
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'Analyse des Métriques - {n_solutions} Solutions de l\'Archive', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.show()
