from typing import Tuple
import pandas as pd

def load_data(videos_path: str,vms_path: str,) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df1 = pd.read_csv(videos_path, sep=';')  #chargement dataset video
    # print(df1.head(10))  

    df2 = pd.read_csv(vms_path, sep=';', encoding='latin-1')  #chargement dataset vm
    # print(df2.head(10)) 

    return df1, df2


load_data('datasets/videos.csv', 'datasets/machines_virtuelles.csv')