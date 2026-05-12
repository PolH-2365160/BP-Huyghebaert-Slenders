import time
import os
import sys
import gc
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn

# --- CONFIGURATIE ---
# Zet op True om de benchmarks opnieuw te draaien.
# Standaard op False: laadt data uit de CSV.
REGENERATE_DATA = False

# Matplotlib instellen voor opslaan zonder pop-ups
matplotlib.use("Agg")
plt.ioff()

# Paden instellen
current_dir = Path(os.path.abspath(os.path.dirname(__file__)))
root_dir = current_dir.parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from Functies.Impliciete_functie import impliciete_oplossing




CSV_PAD = current_dir / 'tijd_analyse_data.csv'
MODEL_PAD = root_dir / 'Neuraal Netwerk' / 'Getrande_modellen' / '138_punten_model.pth'

# Parameters voor de benchmark (alleen bij REGENERATE_DATA = True)
L, T, R, D = 20, 20, 1, 1
HERHALINGEN = 10
BATCH_SIZE = 50000  

AANTAL_PUNTEN = [
    2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 22, 24,
    28, 30, 36, 39, 42, 45, 49, 53, 57, 67, 72, 84, 97, 104,
    120, 138, 148, 169, 181, 206, 220, 235, 250, 283, 301, 340, 361, 383, 406,
    430, 481, 536, 565, 626, 658, 691, 725, 796, 833, 871, 950, 1033, 1076,
    1120, 1211, 1258, 1306, 1355, 1456, 1508, 1561, 1615, 1670, 1726, 1783, 
    1841, 1900, 2021, 2083, 2146, 2275, 2408, 2476, 2545, 2615, 2686, 2758, 
    2831, 2905, 3056, 3133, 3290, 3370, 3451, 3616, 3700, 3785, 3958, 4135, 
    4225, 4316, 4408, 4501, 4700, 5500, 6200, 7000, 8000, 9200, 12000, 14000
][::-1]

def create_model():
    """Definieert de architectuur van het neuraal netwerk."""
    return nn.Sequential(
        nn.Linear(2, 50), nn.Tanh(),
        nn.Linear(50, 50), nn.Tanh(),
        nn.Linear(50, 50), nn.Tanh(),
        nn.Linear(50, 50), nn.Tanh(),
        nn.Linear(50, 50), nn.Tanh(),
        nn.Linear(50, 1),
    )

def format_duration(seconds):
    """Leesbare tijdweergave."""
    return str(timedelta(seconds=int(max(0, seconds))))

def benchmark_modellen(device):
    """Voert de rekentijd-metingen uit voor zowel numeriek als NN."""
    gem_tijd_exact, gem_tijd_nn = [], []
    
    print(f"Model laden op {device}...")
    model = create_model().to(device)
    model.load_state_dict(torch.load(MODEL_PAD, map_location=device, weights_only=True))
    model.eval()

    totale_starttijd = time.time()
    totaal_runs = len(AANTAL_PUNTEN)

    for idx, aantal in enumerate(AANTAL_PUNTEN, start=1):
        run_start = time.time()
        
        # Grid voorbereiden
        x_eval = np.linspace(0, L, aantal)
        t_eval = np.linspace(0, T, aantal + 1)
        X_eval, T_eval = np.meshgrid(x_eval, t_eval)
        input_flat = np.stack((X_eval.flatten(), T_eval.flatten()), axis=1)
        input_tensor = torch.tensor(input_flat, dtype=torch.float32).to(device)

        # 1. Benchmark Numerieke Methode
        tijden_exact = []
        for _ in range(HERHALINGEN):
            s_e = time.time()
            u_exact = [1 / (1 + np.exp(x_eval - 10))]
            for _ in range(aantal):
                u_exact.append(impliciete_oplossing(L, T, R, D, aantal, aantal, u_exact[-1]))
            tijden_exact.append(time.time() - s_e)
        gem_tijd_exact.append(np.mean(tijden_exact))

        # 2. Benchmark Neuraal Netwerk
        tijden_nn = []
        with torch.inference_mode():
            for _ in range(HERHALINGEN):
                s_n = time.time()
                preds = []
                # Batching om GPU Out-of-Memory te voorkomen
                for i in range(0, input_tensor.size(0), BATCH_SIZE):
                    preds.append(model(input_tensor[i:i + BATCH_SIZE]))
                
                # Resultaat ophalen en synchroniseren voor M3 chip
                _ = torch.cat(preds, dim=0).cpu().numpy()
                if device.type == 'mps':
                    torch.mps.synchronize()
                
                tijden_nn.append(time.time() - s_n)
        gem_tijd_nn.append(np.mean(tijden_nn))

        # Geheugen rigoureus opschonen na elke grid-grootte
        del input_tensor, X_eval, T_eval, preds
        if device.type == 'mps': torch.mps.empty_cache()
        gc.collect()

        print(f"[{idx}/{totaal_runs}] Grid: {aantal}x{aantal} | Run: {format_duration(time.time()-run_start)} | Totaal: {format_duration(time.time()-totale_starttijd)}")

    df = pd.DataFrame({'Aantal_punten': AANTAL_PUNTEN, 'Gemiddelde_tijd_exact': gem_tijd_exact, 'Gemiddelde_tijd_nn': gem_tijd_nn})
    df.to_csv(CSV_PAD, index=False)
    return df

def plot_results(data):
    """Genereert de visuele analyses."""

    # --- FIX: Zorg dat de data echt als getallen worden gezien ---
    data['Gemiddelde_tijd_exact'] = pd.to_numeric(data['Gemiddelde_tijd_exact'], errors='coerce')
    data['Gemiddelde_tijd_nn'] = pd.to_numeric(data['Gemiddelde_tijd_nn'], errors='coerce')
    
    # Verwijder eventuele regels waar de conversie mislukte (NaN)
    data = data.dropna(subset=['Gemiddelde_tijd_exact', 'Gemiddelde_tijd_nn'])

    # Plot 1: Rekentijd vs Aantal Punten
    plt.figure(figsize=(10, 6), num=1)
    plt.loglog(data['Aantal_punten'], data['Gemiddelde_tijd_exact'], label='Numerieke methode')
    plt.loglog(data['Aantal_punten'], data['Gemiddelde_tijd_nn'], label='Neuraal Netwerk')
    plt.xlabel('Aantal Punten (Nx = Nt)', fontsize=14)
    plt.ylabel('Gemiddelde Rekentijd (seconden)', fontsize=14)
    plt.title('Rekentijd vs Aantal Punten', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(current_dir / 'Rekentijd_vs_Aantal_Punten.png')

    # Plot 2: Relatieve Snelheidswinst
    data['Snelheidswinst'] = ((data['Gemiddelde_tijd_exact'] / data['Gemiddelde_tijd_nn'])) 
    plt.figure(figsize=(10, 6), num=2)
    plt.semilogx(data['Aantal_punten'], data['Snelheidswinst'], label='Snelheidswinst NN tov Numerieke methode')
    plt.xlabel('Aantal Punten (Nx = Nt)', fontsize=14)
    plt.ylabel('Snelheidswinst', fontsize=14)
    plt.title('Snelheidswinst van Neuraal Netwerk t.o.v. Numerieke methode', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(current_dir / 'Relatieve_Snelheidswinst.png')

    # Plot 3: Absoluut Verschil in Rekentijd
    data['Verschill_tijden'] = data['Gemiddelde_tijd_exact'] - data['Gemiddelde_tijd_nn']
    plt.figure(figsize=(10, 6), num=4)
    plt.loglog(data['Aantal_punten'], data['Verschill_tijden'], label='Absoluut Verschil in Rekentijd')
    plt.xlabel('Aantal Punten (Nx = Nt)', fontsize=14)
    plt.ylabel('Verschil in Rekentijd (seconden)', fontsize=14)
    plt.title('Verschil in Rekentijd tussen Numerieke methode en Neuraal Netwerk', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(current_dir / 'Absoluut_Verschil_in_Rekentijd.png')

    print(f"\nBenchmark voltooid. Gemiddelde winst: {data['Snelheidswinst'].mean():.2f}x")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    if REGENERATE_DATA:
        print("Start nieuwe benchmarks...")
        df = benchmark_modellen(device)
    else:
        if CSV_PAD.exists():
            print(f"Data inladen uit bestaande CSV: {CSV_PAD}")
            df = pd.read_csv(CSV_PAD)
        else:
            print("WAARSCHUWING: CSV niet gevonden. Benchmarks worden geforceerd gestart.")
            df = benchmark_modellen(device)
    
    plot_results(df)
