import numpy as np
from utils import load_data
import pandas as pd
from typing import Tuple

class Donnees:
    #pour recuperer les donnees des datasets et calculer la matrice Uij
    def __init__(self, videos: pd.DataFrame, vms: pd.DataFrame):
        # Vidéos
        self.n = len(videos)
        self.m_i = videos["Memory required (MB)"].to_numpy(dtype=float)
        self.q_i = videos["Input file size (MB)"].to_numpy(dtype=float)
        self.w_i = videos["wi"].to_numpy(dtype=float)
        self.q_out = videos["Output file size (MB)"].to_numpy(dtype=float)
        

        if "type" not in vms.columns:
            raise ValueError(
                "La colonne 'type' est absente dans machines_virtuelles.csv. "
                "Elle est nécessaire pour calculer FUR."
            )

        # Nettoyage du format (au cas où)
        type_col = vms["type"].astype(str).str.strip().str.lower()

        # Bool : True si Fog, False sinon
        self.is_fog_j = type_col.eq("fog").values

        # VMs
        self.p = len(vms)
        self.P_j = vms["cpu_power_MIPS"].to_numpy(dtype=float)
        self.lambda_j = vms["lambda (s)"].to_numpy(dtype=float)
        self.beta_j = vms["beta (MB)"].to_numpy(dtype=float)
        self.gamma_j = vms["gamma (MB)"].to_numpy(dtype=float)
        self.energy_j = vms["P_energy (Watts)"].to_numpy(dtype=float)
        self.Dij = vms["Dij (MBps)"].to_numpy(dtype=float)
        self.distances = vms["distance (km)"].to_numpy(dtype=float)
        self.memory_capacity = vms["memory_capacity (MB)"].to_numpy(dtype=float)
        self.U_ij = self.compute_Uij()

    def compute_Uij(self) -> np.ndarray:
   
        U_ij = np.zeros((self.n, self.p), dtype=float)
        c = 300000 

        for i in range(self.n):
            for j in range(self.p):
                L_transfert = (2 * self.distances[j]) / c
                processing_time = self.w_i[i] / self.P_j[j]
                L_sortie = self.q_i[i] / self.Dij[j]

                U_ij[i, j] = L_transfert + processing_time + L_sortie

        return U_ij

class Solution:
    def __init__(self, assignment, donnees: Donnees):
    
        self.assignment = assignment  # Affectation des vidéos aux VMs
        self.donnees = donnees        # L'instance du problème
        self.makespan = None
        self.cost = None
        self.energy = None

    def evaluate(self):
        n, p = self.donnees.n, self.donnees.p  # n = nombre de vidéos, p = nombre de VMs
        U_ij = self.donnees.U_ij  # Matrice des temps d'exécution entre vidéos et VMs

        # 1. Calcul du makespan (temps de traitement total)
        load = np.zeros(p, dtype=float)  # Charge par machine virtuelle
        for i in range(n):
            vm = self.assignment[i]  # VM à laquelle la vidéo i est assignée
            load[vm] += U_ij[i, vm]  # Charge totale pour cette VM
        # Le makespan est la machine ayant la charge maximale
        self.makespan = load.max()

        # 2. Calcul du coût
        total_cost = 0.0
        for i in range(n):
            j = self.assignment[i]  # VM assignée à la vidéo i
            U = U_ij[i, j]
            m = self.donnees.m_i[i]  # Mémoire utilisée par la vidéo i
            q = self.donnees.q_i[i]  # Taille des données entrantes de la vidéo i
            lambda_j = self.donnees.lambda_j[j]
            beta_j = self.donnees.beta_j[j]
            gamma_j = self.donnees.gamma_j[j]
            total_cost += (lambda_j * U + beta_j * m + gamma_j * q)
        self.cost = total_cost
        
        # 3. Calcul de l'énergie
        total_energy = 0
        for j in range(p):
            total_energy += load[j] * self.donnees.energy_j[j]  # Charge par VM * énergie de la VM

        self.energy = total_energy


