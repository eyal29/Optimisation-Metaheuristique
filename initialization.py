from typing import Tuple
import pandas as pd
import yaml

from algo import ParetoArchive, generate_valid_solution
from utils_to_algo import Donnees

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
    
def load_data(videos_path: str,vms_path: str,) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df1 = pd.read_csv(videos_path, sep=';')  #chargement dataset video
    df2 = pd.read_csv(vms_path, sep=';', encoding='latin-1')  #chargement dataset vm
    return df1, df2

def initialize_full_algorithm(config_path: str = "config.yaml"):
    """
    chargement des données, création de la population initiale, et configuration 
    des objets de suivi et d'archive.
    """
    # 1. Chargement de la configuration
    config = load_config(config_path)
    
    # 2. Chargement des données
    videos_path = config["paths"]["videos"]
    vms_path = config["paths"]["vms"]
    videos, vms = load_data(videos_path, vms_path)
    
    donnees = Donnees(videos, vms)

    # 3. Initialisation de la population et de l'Archive
    POP_SIZE = config["gwo"]["population_size"]
    ARCHIVE_MAX = config["gwo"]["max_archive_size"]
    
    valid_solutions = [generate_valid_solution(donnees) for _ in range(POP_SIZE)]
    archive = ParetoArchive(max_size=ARCHIVE_MAX)
    
    # 4. Paramètres du early stopping
    hv_history = None
    ref_point = None
    iterations_without_improvement = 0
    prev_hv = None

    return donnees, config, valid_solutions, archive, hv_history, ref_point, iterations_without_improvement, prev_hv