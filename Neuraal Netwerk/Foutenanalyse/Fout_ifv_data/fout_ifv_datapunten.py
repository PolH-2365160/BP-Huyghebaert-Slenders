import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
import pandas as pd
import re
import time



# Gebruik pathlib for clearer path handling and ensure Final is on sys.path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parents[2]  # This should point to Final
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from Functies.Impliciete_functie import impliciete_oplossing

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

# Het script blijft volledig binnen Final draaien: alle paden worden relatief aan `current_dir` opgebouwd.
MODEL_DIR = current_dir.parents[2] / 'Neuraal Netwerk' / 'Getrainde_modellen' / 'Modellen_fout_ifv_data'
PLOT_RAW = current_dir / 'fout_ifv_datapunten.png'

# Deze waarden bepalen de resolutie van de foutanalyse.
NX_EVAL = 2000
NT_EVAL = 2000
L = 20
T = 20
R = 1
D = 1



def build_reference_solution(x_eval: np.ndarray) -> np.ndarray:
    """Bereken de numerieke referentie-oplossing op het volledige rooster."""
    u_exact = [1 / (1 + np.exp(x_eval - 10))]
    for _ in range(NT_EVAL):
        volgende_stap = impliciete_oplossing(L, T, R, D, NX_EVAL, NT_EVAL, u_exact[-1])
        u_exact.append(volgende_stap)
    return np.array(u_exact)


def build_input_tensor(device: torch.device) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
    """Maak het rooster voor de modelevaluatie."""
    x_eval = np.linspace(0, L, NX_EVAL)
    t_eval = np.linspace(0, T, NT_EVAL + 1)
    x_grid, t_grid = np.meshgrid(x_eval, t_eval)
    input_eval = np.stack((x_grid.flatten(), t_grid.flatten()), axis=1)
    input_eval_tensor = torch.tensor(input_eval, dtype=torch.float32).to(device)
    return x_eval, input_eval_tensor, build_reference_solution(x_eval)


def find_model_paths() -> list[Path]:
    """Zoek alle modelbestanden die bij deze foutanalyse horen."""
    return sorted(MODEL_DIR.glob('**/*_punten_model.pth'))


def parse_model_metadata(pad: Path) -> tuple[int, int] | None:
    """Lees seed en aantal trainingsdatapunten uit de bestandsnaam."""
    match = re.search(r'(\d+)_punten_model\.pth', pad.name)
    seed_match = re.search(r'modellen_3000_epochs_seed_(\d+)', pad.parent.name)
    if not match or not seed_match:
        return None
    seed = int(seed_match.group(1))
    aantal_datapunten = int(match.group(1))
    return seed, aantal_datapunten


def evaluate_models(alle_paden: list[Path], input_eval_tensor: torch.Tensor, u_exact_array: np.ndarray, device: torch.device) -> pd.DataFrame:
    """Laad elk model, bereken de fout en verzamel de resultaten in een dataframe."""
    resultaten = []
    model = create_model().to(device)
    globale_starttijd = time.time()

    for count, pad in enumerate(alle_paden):
        print(f"{count + 1}/{len(alle_paden)}")
        if count == 0:
            print("[Timing] Eerste model gestart. Totale tijdsschatting volgt na model 1.")
        else:
            verstreken = time.time() - globale_starttijd
            gemiddelde_tijd = verstreken / count
            resterende_tijd = gemiddelde_tijd * (len(alle_paden) - count)
            totale_schatting = verstreken + resterende_tijd
            print(
                f"[Timing] Verstreken: {format_tijd(verstreken)} | "
                f"Resterend (geschat): {format_tijd(resterende_tijd)} | "
                f"Totale tijd (geschat): {format_tijd(totale_schatting)}"
            )

        metadata = parse_model_metadata(pad)
        if metadata is None:
            print(f"Bestandsnaam {pad.name} komt niet overeen met het verwachte patroon. Overslaan.")
            continue

        seed, aantal_datapunten = metadata
        model_starttijd = time.time()

        # Laad het getrainde model en zet het in evaluatiemodus.
        model.load_state_dict(torch.load(pad, map_location=device))
        model.eval()

        with torch.no_grad():
            u_pred_flat = model(input_eval_tensor).cpu().numpy().flatten()
        u_pred = u_pred_flat.reshape(NT_EVAL + 1, NX_EVAL)

        l2_error_rel = np.linalg.norm(u_pred - u_exact_array) / np.linalg.norm(u_exact_array)
        l2_error_abs = np.linalg.norm(u_pred - u_exact_array)

        resultaten.append({
            'Aantal_datapunten': aantal_datapunten,
            'L2 Error Relatief': l2_error_rel,
            'L2 Error Absoluut': l2_error_abs,
            'Seed': seed,
        })

        model_duur = time.time() - model_starttijd
        verstreken_na = time.time() - globale_starttijd
        afgerond = count + 1
        resterende_tijd_na = (verstreken_na / afgerond) * (len(alle_paden) - afgerond)
        print(
            f"[Timing] Model duur: {format_tijd(model_duur)} | "
            f"Tot nu toe: {format_tijd(verstreken_na)} | "
            f"Resterend (geschat): {format_tijd(resterende_tijd_na)}"
        )

    df = pd.DataFrame(resultaten)
    if not df.empty:
        df = df.sort_values(by='Aantal_datapunten')
    else:
        print("Let op: er is geen data weggeschreven. Controleer of de regex de bestandsnamen goed leest.")

    print(f"Totale verwerkingstijd: {format_tijd(time.time() - globale_starttijd)}")
    return df


def remove_outliers(df: pd.DataFrame, kolom: str, factor: float = 2) -> pd.DataFrame:
    """Verwijder uitschieters met de IQR-methode."""
    q1 = df[kolom].quantile(0.25)
    q3 = df[kolom].quantile(0.75)
    iqr = q3 - q1
    ondergrens = q1 - factor * iqr
    bovengrens = q3 + factor * iqr
    return df[(df[kolom] >= ondergrens) & (df[kolom] <= bovengrens)]


def keep_groups_with_min_size(df: pd.DataFrame, kolom: str, min_aantal: int = 2) -> pd.DataFrame:
    """Laat alleen groepen over waarvoor minstens `min_aantal` modellen beschikbaar zijn."""
    return df.groupby(kolom).filter(lambda groep: len(groep) >= min_aantal)


def prepare_grouped_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_niet_gegroepeerd = df.copy()
    df_gefilterd = remove_outliers(df.copy(), 'L2 Error Relatief')
    df_gefilterd = keep_groups_with_min_size(df_gefilterd, 'Aantal_datapunten', 2)

    # NIEUW: Bewaar de gefilterde, maar nog niet gegroepeerde data voor de scatterplot
    df_gefilterd_ongegroepeerd = df_gefilterd.copy()

    df_gefilterd_groep = (
        df_gefilterd.groupby('Aantal_datapunten', as_index=False)
        .agg(
            Aantal_modellen=('L2 Error Relatief', 'count'),
            L2_Error_Relatief_Gemiddeld=('L2 Error Relatief', 'mean'),
            L2_Error_Relatief_Std=('L2 Error Relatief', 'std'),
        )
        .sort_values(by='Aantal_datapunten')
    )

    df_gegroepeerd = (
        df_niet_gegroepeerd.groupby('Aantal_datapunten', as_index=False)
        .agg(
            Aantal_modellen=('L2 Error Relatief', 'count'),
            L2_Error_Relatief_Gemiddeld=('L2 Error Relatief', 'mean'),
            L2_Error_Relatief_Std=('L2 Error Relatief', 'std'),
        )
        .sort_values(by='Aantal_datapunten')
    )

    df_gegroepeerd['L2_Error_Relatief_Std'] = df_gegroepeerd['L2_Error_Relatief_Std'].fillna(0.0)
    df_gefilterd_groep['L2_Error_Relatief_Std'] = df_gefilterd_groep['L2_Error_Relatief_Std'].fillna(0.0)

    # Geef nu 4 elementen terug
    return df_niet_gegroepeerd, df_gefilterd_ongegroepeerd, df_gefilterd_groep, df_gegroepeerd


def plot_results(df_scatter: pd.DataFrame, df_gefilterd: pd.DataFrame) -> None:
    """Maak de twee overzichtsplots en sla ze op in `current_dir`."""
    x = df_gefilterd['Aantal_datapunten'].to_numpy()
    y = df_gefilterd['L2_Error_Relatief_Gemiddeld'].to_numpy()
    y_std = df_gefilterd['L2_Error_Relatief_Std'].to_numpy()
    y_lower = np.maximum(0, y - y_std)
    y_upper = y + y_std

    plt.figure(figsize=(11, 6), num=1)
    plt.scatter(
        df_scatter['Aantal_datapunten'], # Hier gebruikten we eerst df_niet_gegroepeerd
        df_scatter['L2 Error Relatief'],
        s=12,
        alpha=0.22,
        label='Fout Neuraal Netwerk per Model',
    )
    plt.plot(x, y, color='tab:red', linewidth=2, label='Gemiddelde Fout per Datapunt', linestyle='-')
    plt.fill_between(x, y_lower, y_upper, color='tab:red', alpha=0.18, label='$\\pm$1 standaarddeviatie')
    plt.xscale('log')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Aantal trainingsdatapunten', fontsize=14)
    plt.ylabel('Relatieve $L^2$-fout', fontsize=14)
    plt.title('Relatieve $L^2$-fout Verschillende Modellen i.f.v. Aantal Trainingsdatapunten', fontsize=16)
    plt.grid(True, which='both', alpha=0.25)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOT_RAW, dpi=300)


def main() -> None:
    """Draai de volledige foutanalyse voor de getrainde modellen."""
    alle_paden = find_model_paths()
    print(f"{len(alle_paden)} modellen gevonden in {MODEL_DIR}")
    if not alle_paden:
        print("Geen modelbestanden gevonden; stop zonder verdere verwerking.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    x_eval, input_eval_tensor, u_exact_array = build_input_tensor(device)
    df_raw = evaluate_models(alle_paden, input_eval_tensor, u_exact_array, device)
    if df_raw.empty:
        print("Geen geldige modelresultaten om te plotten.")
        return

    # Vang de gefilterde scatter data op in df_scatter
    _, df_scatter, df_gefilterd, _ = prepare_grouped_data(df_raw)
    
    # Geef de juiste data door aan plot_results
    plot_results(df_scatter, df_gefilterd)
    plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
