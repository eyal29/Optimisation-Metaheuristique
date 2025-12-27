# Projet Métaheuristique Hybride GWO-NSGA-II

Ce projet implémente un algorithme hybride combinant le **Grey Wolf Optimizer (GWO)** et **NSGA-II** pour résoudre un problème d'optimisation multi-objectifs. L'objectif est de traiter des vidéos à l'aide de machines virtuelles (VMs) tout en minimisant plusieurs critères : **Makespan**, **Coût d'Exécution** et **Consommation d'Énergie**.

## Prérequis

1. **Python 3.6+**
2. **Bibliothèques Python** :
   - NumPy
   - Pandas
   - Matplotlib
   - PyYAML
   - mplcursors

## Installation

1. **Clonez ce dépôt** :
   ```bash
   git clone https://github.com/eyal29/Optimisation-Metaheuristique.git
   cd Optimisation-Metaheuristique
2. **Installer les dépendances** :  
Installez les bibliothèques nécessaires via le fichier requirements.txt :
    ```bash
    pip install -r requirements.txt    
3. **Exécuter le projet** :
Une fois les dépendances installées, vous pouvez lancer le projet via le fichier principal main.py :
    ```bash
    python main.py
## Structure du Projet
    
    ├── algo_hybride/
    │   ├── algo.py              # Logique de l'algorithme hybride GWO-NSGA-II
    │   ├── initialization.py    # Initialisation des données et objets
    │   ├── utils_to_algo.py     # Modèles et utilitaires
    ├── analyses/
    │   ├── metrics.py           # Calcul des métriques de performance
    │   ├── visualization.py     # Visualisation des résultats
    │   └── experiments.py       # Exécution des expérimentations
    ├── heuristiques/            # Algorithmes de référence pour les comparaisons
    |   ├── greedy.py          
    │   ├── gwo_simple.py    
    │   └── nsga2.py    
    ├── datasets/                #Datasets de VM et Vidéos
    |   ├── machines_virtuelles.py          
    │   ├── videos.py  
    ├── config.yaml              # Fichier de configuration pour l'algorithme      
    └── main.py                  # Point d'entrée du programme

## Temps d'execution
L'exécution de l'algorithme hybride prend environ 1 minute. Il faut ensuite attendre environ 5 minutes pour la comparaison avec les autres algorithmes et l'affichage des résultats.


## Auteurs
- Projet réalisé par Eya LACHHEB, Lina ERRADI, Sara FLEGINES, Lyliane HADIOUCHE


