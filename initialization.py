from typing import Tuple
import pandas as pd
import yaml

from algo import generate_valid_solution
from utils_to_algo import Donnees

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
    
def load_data(videos_path: str,vms_path: str,) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df1 = pd.read_csv(videos_path, sep=';')  #chargement dataset video
    df2 = pd.read_csv(vms_path, sep=';', encoding='latin-1')  #chargement dataset vm
    return df1, df2

def initialize_algorithm(config_path: str = "config.yaml"):
    """
    Initialisation de l'algorithme GWO : chargement des configurations et des données.
    
    Retourne :
        - donnees : Objet contenant les données du problème
        - config : Configuration lue depuis le fichier
    """
    # Chargement de la configuration
    config = load_config(config_path)
    
    # Chargement des données
    videos_path = config["paths"]["videos"]
    vms_path = config["paths"]["vms"]
    videos, vms = load_data(videos_path, vms_path)
    
    donnees = Donnees(videos, vms)

    # Initialisation de la population
    POP_SIZE = config["gwo"]["population_size"]
    valid_solutions = [generate_valid_solution(donnees) for _ in range(POP_SIZE)]

    return donnees, config, valid_solutions