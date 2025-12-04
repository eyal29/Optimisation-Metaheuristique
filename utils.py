from typing import Tuple
import pandas as pd
import yaml

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
    
def load_data(videos_path: str,vms_path: str,) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df1 = pd.read_csv(videos_path, sep=';')  #chargement dataset video
    df2 = pd.read_csv(vms_path, sep=';', encoding='latin-1')  #chargement dataset vm
    return df1, df2
