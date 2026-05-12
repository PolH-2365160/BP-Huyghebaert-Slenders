import os
import sys
from pathlib import Path
from typing_extensions import Final
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
import pandas as pd
import re
import time



# Zorg ervoor dat de root directory in het path staat voor imports
current_dir = Path(os.path.abspath(os.path.dirname(__file__)))
root_dir = current_dir.parents[2]  # Twee niveaus omhoog (equivalent aan '../../')
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from Functies.Impliciete_functie import impliciete_oplossing

# --- INSTELLINGEN & PADEN ---
PAD_MAP_MODELLEN = root_dir / 'Neuraal Netwerk' / 'Getrainde_modellen' / 'Modellen_fout_ifv_trainingsiteraties'

PLOT_PAD = current_dir / 'fout_ifv_trainingsiteraties.png'


# Parameters
NX_EVAL = 2000
NT_EVAL = 2000
L = 20
T = 20
R = 1
D = 1


def create_model():
    """Definieert en retourneert de architectuur van het neurale netwerk."""
    return nn.Sequential(
        nn.Linear(2, 50), nn.Tanh(),
        nn.Linear(50, 50), nn.Tanh(),
        nn.Linear(50, 50), nn.Tanh(),
        nn.Linear(50, 50), nn.Tanh(),
        nn.Linear(50, 50), nn.Tanh(),
        nn.Linear(50, 1),
    )


def format_tijd(seconden):
    seconden = max(0, int(seconden))
    uren, rest = divmod(seconden, 3600)
    minuten, seconden = divmod(rest, 60)
    return f"{uren:02d}:{minuten:02d}:{seconden:02d}"


def build_exact_solution(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Maakt het evaluatiegrid en berekent de exacte oplossing (geoptimaliseerd voor GPU)."""
    x_eval = np.linspace(0, L, NX_EVAL)
    t_eval = np.linspace(0, T, NT_EVAL + 1)
    X_eval, T_eval = np.meshgrid(x_eval, t_eval)

    X_eval_flat = X_eval.flatten()
    T_eval_flat = T_eval.flatten()
    input_eval = np.stack((X_eval_flat, T_eval_flat), axis=1)
    input_eval_tensor = torch.tensor(input_eval, dtype=torch.float32).to(device)

    # Exacte numerieke oplossing berekenen
    u_exact = [1 / (1 + np.exp(x_eval - 10))]
    for _ in range(NT_EVAL):
        volgende_stap = impliciete_oplossing(L, T, R, D, NX_EVAL, NT_EVAL, u_exact[-1])
        u_exact.append(volgende_stap)
    
    u_exact_array = np.array(u_exact)
    u_exact_tensor = torch.tensor(u_exact_array, dtype=torch.float32, device=device)
    norm_exact_tensor = torch.linalg.norm(u_exact_tensor)

    return input_eval_tensor, u_exact_tensor, norm_exact_tensor


def evaluate_models(input_eval_tensor: torch.Tensor, u_exact_tensor: torch.Tensor, norm_exact_tensor: torch.Tensor, device: torch.device) -> pd.DataFrame:
    """Evalueert alle modellen in de map en retourneert de ruwe data."""
    model = create_model().to(device)
    resultaten = []
    
    alle_paden = sorted(list(PAD_MAP_MODELLEN.glob('Modellen_700_data_seed*/*_epoches_model.pth')))
    print(f"{len(alle_paden)} modellen gevonden in {PAD_MAP_MODELLEN}")
    if not alle_paden:
        return pd.DataFrame()

    globale_starttijd = time.time()
    regex = r'(\d+)_epoches_model\.pth'

    for count, pad in enumerate(alle_paden):
        print(f"{count + 1}/{len(alle_paden)}")
        if count > 0:
            verstreken = time.time() - globale_starttijd
            gemiddelde_tijd = verstreken / count
            resterend_aantal = len(alle_paden) - count
            resterende_tijd = gemiddelde_tijd * resterend_aantal
            totale_schatting = verstreken + resterende_tijd
            print(
                f"[Timing] Verstreken: {format_tijd(verstreken)} | "
                f"Resterend (geschat): {format_tijd(resterende_tijd)} | "
                f"Totale tijd (geschat): {format_tijd(totale_schatting)}"
            )
        else:
            print("[Timing] Eerste model gestart. Totale tijdsschatting volgt na model 1.")

        model_starttijd = time.time()
        
        match = re.search(regex, pad.name)
        seed_match = re.search(r'Modellen_700_data_seed(\d+)', pad.parent.name)
        if match and seed_match:
            seed = int(seed_match.group(1))
            epochs = int(match.group(1))
        else:
            print(f"Bestandsnaam {pad.name} komt niet overeen met het verwachte patroon. Overslaan.")
            continue
            
        # Model inladen
        model.load_state_dict(torch.load(pad, map_location=device, weights_only=True))
        model.eval()
        
        with torch.no_grad():
            u_pred = model(input_eval_tensor).view(NT_EVAL + 1, NX_EVAL)
            verschil = u_pred - u_exact_tensor
            l2_error_abs_tensor = torch.linalg.norm(verschil)
            l2_error_rel_tensor = l2_error_abs_tensor / norm_exact_tensor
            
            l2_error_abs = l2_error_abs_tensor.item()
            l2_error_rel = l2_error_rel_tensor.item()
        
        resultaten.append({
            'Epoch': epochs, 
            'L2 Error Relatief': l2_error_rel, 
            'L2 Error Absoluut': l2_error_abs,
            'Seed': seed
        })

        model_duur = time.time() - model_starttijd
        verstreken_na = time.time() - globale_starttijd
        afgerond = count + 1
        gemiddelde_tijd_na = verstreken_na / afgerond
        resterend_aantal_na = len(alle_paden) - afgerond
        resterende_tijd_na = gemiddelde_tijd_na * resterend_aantal_na
        print(
            f"[Timing] Model duur: {format_tijd(model_duur)} | "
            f"Tot nu toe: {format_tijd(verstreken_na)} | "
            f"Resterend (geschat): {format_tijd(resterende_tijd_na)}"
        )

    df = pd.DataFrame(resultaten)
    if not df.empty:
        df = df.sort_values(by='Epoch')
        
    print(f"Totale verwerkingstijd: {format_tijd(time.time() - globale_starttijd)}")
    return df


def uitschieters_verwijderen(df: pd.DataFrame, kolom: str, factor: float = 2) -> pd.DataFrame:
    q1 = df[kolom].quantile(0.25)
    q3 = df[kolom].quantile(0.75)
    iqr = q3 - q1
    ondergrens = q1 - factor * iqr
    bovengrens = q3 + factor * iqr
    return df[(df[kolom] >= ondergrens) & (df[kolom] <= bovengrens)]


def datapunten_met_te_weinig_modellen_verwijderen(df: pd.DataFrame, kolom: str, min_aantal: int = 2) -> pd.DataFrame:
    return df.groupby(kolom).filter(lambda groep: len(groep) >= min_aantal)


def prepare_grouped_data(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filtert en groepeert de dataset en bewaart de nodige CSV's."""
    df_niet_gegroepeerd = df_raw.copy()
    
    # 1. Filter out uitschieters en te kleine groepen
    df_gefilterd = uitschieters_verwijderen(df_raw.copy(), "L2 Error Relatief")
    df_gefilterd = datapunten_met_te_weinig_modellen_verwijderen(df_gefilterd, "Epoch", 2)

    # 2. Bewaar de schone losse punten voor de scatterplot (zonder uitschieters!)
    df_scatter = df_gefilterd.copy()

    # 3. Groepeer de gefilterde data
    df_gefilterd_groep = (
        df_gefilterd.groupby("Epoch", as_index=False)
        .agg(
            Aantal_modellen=("L2 Error Relatief", "count"),
            L2_Error_Relatief_Gemiddeld=("L2 Error Relatief", "mean"),
            L2_Error_Relatief_Std=("L2 Error Relatief", "std"),
        )
        .sort_values(by="Epoch")
    )

    # 4. Groepeer de ongefilterde data (voor de 'gesorteerd' backup)
    df_gegroepeerd = (
        df_niet_gegroepeerd.groupby("Epoch", as_index=False)
        .agg(
            Aantal_modellen=("L2 Error Relatief", "count"),
            L2_Error_Relatief_Gemiddeld=("L2 Error Relatief", "mean"),
            L2_Error_Relatief_Std=("L2 Error Relatief", "std"),
        )
        .sort_values(by="Epoch")
    )

    # Vul NaN std-waarden op met 0.0
    df_gegroepeerd['L2_Error_Relatief_Std'] = df_gegroepeerd['L2_Error_Relatief_Std'].fillna(0.0)
    df_gefilterd_groep['L2_Error_Relatief_Std'] = df_gefilterd_groep['L2_Error_Relatief_Std'].fillna(0.0)

    
    return df_scatter, df_gefilterd_groep


def plot_results(df_scatter: pd.DataFrame, df_gefilterd_groep: pd.DataFrame) -> None:
    """Genereert en bewaart de plots op basis van de opgeschoonde data."""
    x = df_gefilterd_groep['Epoch'].to_numpy() 
    y = df_gefilterd_groep['L2_Error_Relatief_Gemiddeld'].to_numpy()
    y_std = df_gefilterd_groep['L2_Error_Relatief_Std'].to_numpy()
    
    y_lower = np.maximum(0, y - y_std)
    y_upper = y + y_std

    # --- PLOT 1: Scatterplot & Gemiddelde Lijn ---
    plt.figure(figsize=(11, 6), num=1)
    plt.scatter(
        df_scatter['Epoch'], 
        df_scatter['L2 Error Relatief'], 
        s=12, 
        alpha=0.22, 
        label='Fout Neuraal Netwerk per Model'
    )
    plt.plot(x, y, color='tab:red', linewidth=2, label='Gemiddelde Fout per Trainingsiteratie', linestyle='-')
    plt.fill_between(x, y_lower, y_upper, color='tab:red', alpha=0.18, label=r'$\pm$1 standaarddeviatie')
    
    plt.xscale('log')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Aantal trainingsiteraties', fontsize=14)
    plt.ylabel('Relatieve $L^2$-fout', fontsize=14)
    plt.title('Relatieve $L^2$-fout Verschillende Modellen i.f.v. Trainingsiteraties', fontsize=16)
    plt.grid(True, which='both', alpha=0.25)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOT_PAD, dpi=300)




def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Start evaluatie op device: {device}")
    
    # 1. Setup Grid en Exacte Oplossing
    input_eval_tensor, u_exact_tensor, norm_exact_tensor = build_exact_solution(device)
    
    # 2. Bereken alle fouten (dit genereert en bewaart de eerste ruwe CSV)
    df_raw = evaluate_models(input_eval_tensor, u_exact_tensor, norm_exact_tensor, device)
    
    if df_raw.empty:
        print("Geen data beschikbaar om te plotten. Controleer de map en modelbestanden.")
        return

    # 3. Filter uitschieters en groepeer data
    df_scatter, df_gefilterd_groep = prepare_grouped_data(df_raw)
    
    # 4. Teken de figuren en bewaar als PNG
    plot_results(df_scatter, df_gefilterd_groep)
    
    # figuren netjes sluiten in het geheugen
    plt.show()
    plt.close('all')
    print("Alle acties succesvol afgerond.")


if __name__ == '__main__':
    main()