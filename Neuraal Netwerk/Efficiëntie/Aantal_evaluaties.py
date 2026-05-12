import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from pathlib import Path

# --- CONFIGURATIE ---
AANTAL_PUNTEN_TRAIN = 138
HERHALINGEN_TRAIN = 5
EPOCHS = 10000
BATCH_SIZE = 100000
TRAIN_OPNIEUW = True  # Zet op False als je de trainingstijd handmatig wilt opgeven

# Paden instellen
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = Path(os.path.abspath(os.path.join(CURRENT_DIR, "../../")))


CSV_DATA_PAD = CURRENT_DIR / 'tijd_analyse_data.csv'

# Root toevoegen aan sys.path voor custom imports
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Custom imports
from Functies.Pinn_adaptive_weights_functie import Pinn_adaptive_weights

def format_tijd(seconden):
    """Formatteert seconden naar HH:MM:SS."""
    seconden = max(0, int(seconden))
    uren, rest = divmod(seconden, 3600)
    minuten, seconden = divmod(rest, 60)
    return f"{uren:02d}:{minuten:02d}:{seconden:02d}"

def benchmark_training(aantal_punten, epochs, n_herhalingen):
    """Meet hoe lang het duurt om het model te trainen."""
    tijden = []
    print(f"Start training benchmark ({n_herhalingen} herhalingen van {epochs} epochs)...")
    
    for i in range(n_herhalingen):
        start = time.time()
        # Voer de training uit
        Pinn_adaptive_weights(basis_pad=CURRENT_DIR, aantal_trainingspunten=aantal_punten, EPOCHS=epochs)
        duur = time.time() - start
        tijden.append(duur)
        print(f"  Herhaling {i+1}/{n_herhalingen}: {format_tijd(duur)}")
        
    gemiddelde = np.mean(tijden)
    print(f"\nGemiddelde trainingstijd: {gemiddelde:.2f}s ({format_tijd(gemiddelde)})")
    return gemiddelde

def plot_break_even(gem_trainingstijd, csv_pad):
    """Berekent en plot het break-even punt tussen numeriek en NN."""
    if not csv_pad.exists():
        print(f"Fout: {csv_pad} niet gevonden. Run eerst de tijd_analyse benchmark.")
        return

    # Data inladen
    data = pd.read_csv(csv_pad)
    
    # Forceer numerieke types (voorkomt de Arrow/TypeError)
    data['Gemiddelde_tijd_exact'] = pd.to_numeric(data['Gemiddelde_tijd_exact'], errors='coerce')
    data['Gemiddelde_tijd_nn'] = pd.to_numeric(data['Gemiddelde_tijd_nn'], errors='coerce')
    data = data.dropna()

    # Berekening: Hoe vaak moet je het NN draaien om de trainingstijd terug te verdienen?
    # Formule: Aantal keer = Trainingstijd / (Tijd_Exact - Tijd_NN)
    # We gebruiken een min-teken omdat (Tijd_NN - Tijd_Exact) meestal negatief is (NN is sneller)
    data['Break_Even_Iteraties'] = -gem_trainingstijd / (data['Gemiddelde_tijd_nn'] - data['Gemiddelde_tijd_exact'])

    # Plotten
    plt.figure(figsize=(10, 6))
    plt.plot(data['Aantal_punten'], data['Break_Even_Iteraties'], label = 'Aantal Evaluaties')
    plt.yscale('symlog', linthresh=1e-1) 
    plt.xscale('log')
    plt.xlabel('Aantal Punten (Nx = Nt)', fontsize=14)
    plt.ylabel('Aantal Evaluaties', fontsize=14)
    plt.title('Aantal Evaluaties nodig om Trainingstijd te Compenseren', fontsize=16)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.legend(fontsize=14)
    
    save_pad = CURRENT_DIR / 'aantal_evaluaties_om_trainingstijd_te_compenseren.png'
    plt.savefig(save_pad, dpi=300)
    print(f"Grafiek opgeslagen als: {save_pad}")
    plt.show()

def main():
    # 1. Bepaal de trainingstijd
    if TRAIN_OPNIEUW:
        gem_tijd = benchmark_training(AANTAL_PUNTEN_TRAIN, EPOCHS, HERHALINGEN_TRAIN)
    else:
        # Handmatige waarde als je niet telkens 5x10.000 epochs wilt wachten
        gem_tijd = 360.0  # Voorbeeldwaarde in seconden
        print(f"Gebruik handmatige trainingstijd: {gem_tijd}s")

    # 2. Analyseer en plot break-even punt
    plot_break_even(gem_tijd, CSV_DATA_PAD)

if __name__ == "__main__":
    main()