from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


# Maak interne projectimports beschikbaar.
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


try:
    from Functies.Pinn_functie_zonder_adaptive import Pinn_zonder_adaptive_weights
except ImportError:
    print("Fout: Kan 'Pinn_zonder_adaptive_weights' niet vinden. Controleer je sys.path.")
    raise

try:
    from Functies.Pinn_adaptive_weights_functie import Pinn_adaptive_weights
except ImportError:
    print("Fout: Kan 'Pinn_adaptive_weights' niet vinden. Controleer je sys.path.")
    raise


# Bestandslocaties en parameters.

CSV_PAD = current_dir / 'leersnelheid_parameters.csv'
LEERSNELHEID_PLOT_PAD = current_dir / 'leersnelheid_plot.png'
HEATMAP_PAD = current_dir / 'heatmap_adaptive_weights_versus_zonder_adaptive_weights.png'

MODEL_ADAPTIVE_PAD = current_dir / 'model_met_dynamische_gewichten.pth'
MODEL_ZONDER_PAD = current_dir / 'model_zonder_dynamische_gewichten.pth'
MODEL_SLECHT_PAD = current_dir / 'model_zonder_dynamische_gewichten_en_slechte_gewichten.pth'

L = 20.0
T = 20.0
R = 1.0
D = 1
AANTAL_X = 1000
AANTAL_T = 1000


def generate_model():
    """Bouw het netwerk dat voor de vergelijking wordt geladen."""
    return nn.Sequential(
        nn.Linear(2, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 1),
    )


def extract_leersnelheid(result, naam):
    """Haal de eerste drie returnwaarden op en geef een duidelijke fout als er niets terugkomt.

    Retourneert (k_history, k_inf, alpha, model_or_None).
    """
    if result is None:
        raise RuntimeError(f"{naam} gaf geen resultaat terug.")
    # Verwacht minimaal (K_history, K_inf, alpha). Optioneel kan het model als vierde element terugkomen.
    if isinstance(result, tuple) or isinstance(result, list):
        if len(result) < 3:
            raise RuntimeError(f"{naam} gaf minder dan drie waarden terug.")
        k_history = result[0]
        k_inf = result[1]
        alpha = result[2]
        model = result[3] if len(result) > 3 else None
        return k_history, k_inf, alpha, model
    raise RuntimeError(f"{naam} retourneerde een onverwacht resultaattype: {type(result)}")


def naar_float(waarde):
    """Zet een tensor, lijst of array om naar een enkele float waar mogelijk."""
    if isinstance(waarde, torch.Tensor):
        waarde = waarde.detach().cpu().numpy()
    if isinstance(waarde, (list, tuple, np.ndarray, pd.Series)):
        waarde = np.asarray(waarde).reshape(-1)[0]
    return float(waarde)


def laad_leersnelheid_gegevens():
    """Laad de leersnelheidswaarden voor de drie modelvarianten.

    Nu vraagt deze functie expliciet ook om de getrainde modellen terug te krijgen
    (door `returns=['Leersnelheid', 'model']` te gebruiken bij de PINN-functies).
    """
    # Dynamische gewichten.
    resultaat_adaptive = Pinn_adaptive_weights(
        basis_pad=current_dir,
        naam_model='',
        plot_comparison_loss_name='comparison_loss_dynamische_gewichten',
        returns=['Leersnelheid', 'model'],
        N_avg=10,
    )
    k_history_adaptive, k_infinity_adaptive, alpha_adaptive, model_adaptive = extract_leersnelheid(
        resultaat_adaptive,
        'Pinn_adaptive_weights',
    )

    # Zonder dynamische gewichten, maar met goede gewichten.
    resultaat_zonder = Pinn_zonder_adaptive_weights(
        basis_pad=current_dir,
        naam_model='',
        plot_comparison_loss_name='comparison_loss_zonder_dynamische_gewichten',
        returns=['Leersnelheid', 'model'],
        lam_bdy=1,
        lam_data=10,
        lam_init=40,
        lam_pde=5,
    )
    k_history, k_infinity, alpha, model_zonder = extract_leersnelheid(
        resultaat_zonder,
        'Pinn_zonder_adaptive_weights',
    )

    # Zonder dynamische gewichten, met ongunstige hyperparameters.
    resultaat_slecht = Pinn_zonder_adaptive_weights(
        basis_pad=current_dir,
        naam_model='',
        returns=['Leersnelheid', 'model'],
        lam_bdy=1000,
        lam_data=10,
        lam_init=500,
        lam_pde=1,
    )
    k_history_slecht, k_infinity_slecht, alpha_slecht, model_slecht = extract_leersnelheid(
        resultaat_slecht,
        'Pinn_zonder_adaptive_weights (slecht)',
    )

    return (
        k_history_adaptive,
        k_infinity_adaptive,
        alpha_adaptive,
        model_adaptive,
        k_history,
        k_infinity,
        alpha,
        model_zonder,
        k_history_slecht,
        k_infinity_slecht,
        alpha_slecht,
        model_slecht,
    )


def schrijf_leersnelheid_csv(waarden):
    """Schrijf de opgehaalde leersnelheidswaarden weg naar CSV."""
    namen = [
        'K_history_adaptive', 'K_infinity_adaptive', 'alpha_adaptive',
        'K_history', 'K_infinity', 'alpha',
        'K_history_slecht', 'K_infinity_slecht', 'alpha_slecht',
    ]

    csv_data = {}
    for naam, waarde in zip(namen, waarden):
        if isinstance(waarde, (torch.Tensor, list, tuple, np.ndarray, pd.Series)):
            csv_data[naam] = pd.Series(np.asarray(waarde).reshape(-1))
        else:
            csv_data[naam] = pd.Series([waarde])

    pd.DataFrame(csv_data).to_csv(CSV_PAD, index=False)


# def format_alpha(waarde: float) -> str:
#     """Formatteer alpha met komma als decimaalteken.

#     Args:
#         waarde (float): Kommagetal met .

#     Returns:
#         str: Kommagetal met ,
#     """
#     return f"{waarde:.3f}".replace('.', ',')


def leer_curve(t: np.ndarray, k_inf: float, alpha: float) -> np.ndarray:
    """Bereken de leersnelheid volgens een exponentiële curve.

    Args:
        t (np.ndarray): Tijdwaarden.
        k_inf (float): Limiet van de leersnelheid.
        alpha (float): Leersnelheidsparameter.

    Returns:
        np.ndarray: De leersnelheid voor elke tijdwaarde.
    """
    return k_inf * (1 - np.exp(-alpha * t))


def plot_leersnelheid(t: np.ndarray, k_adaptive: np.ndarray, k_zonder_adaptive: np.ndarray, k_slecht: np.ndarray, alpha_adaptive: float, alpha: float, alpha_slecht: float):
    """Plot de leersnelheid voor de drie configuraties.

    Args:
        t (np.ndarray): Tijdwaarden.
        k_adaptive (np.ndarray): De leersnelheid voor de adaptieve gewichten.
        k_zonder_adaptive (np.ndarray): De leersnelheid voor de zonder adaptieve gewichten.
        k_slecht (np.ndarray): De leersnelheid voor de slechte gewichten.
        alpha_adaptive (float): De leersnelheidsparameter voor de adaptieve gewichten.
        alpha (float): De leersnelheidsparameter voor de zonder adaptieve gewichten.
        alpha_slecht (float): De leersnelheidsparameter voor de slechte gewichten.
    """
    plt.figure(figsize=(10, 6), num=1)
    plt.plot(t, k_adaptive, label=f'Dynamische Gewichten (α = {alpha_adaptive:.3f})', color='green', linewidth=3)
    plt.plot(t, k_zonder_adaptive, label=f'Goed Gekozen Gewichten (α = {alpha:.3f})', color='red', linewidth=3, linestyle='-.')
    plt.plot(t, k_slecht, label=f'Slecht Gekozen Gewichten (α = {alpha_slecht:.3f})', color='orange', linewidth=3, linestyle='--')
    plt.axhline(y=1.0, color='grey', linestyle='--', alpha=0.5, label='Max ($K_{\\infty} = 1.0$)')
    plt.xlabel('Trainingsiteratie', fontsize=14)
    plt.ylabel('K(E)', fontsize=14)
    plt.xticks(np.arange(0, 16, 2), fontsize=14)
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
    plt.title('Invloed Dynamische Gewichten-Algoritme op Leersnelheid', fontsize=16)
    plt.ylim(0, 1.05)
    plt.xlim(0, 15)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.savefig(LEERSNELHEID_PLOT_PAD, dpi=300)


def laad_modellen()-> tuple[nn.Sequential, nn.Sequential]:
    """Laad de twee modellen die in de heatmapvergelijking gebruikt worden.

    Returns:
        tuple[nn.Sequential, nn.Sequential]: Een tuple met het model met dynamische gewichten en zonder dynamische gewichten
    """
    model_adaptive_weights = generate_model()
    model_adaptive_weights.load_state_dict(torch.load(MODEL_ADAPTIVE_PAD, map_location='cpu'))
    model_adaptive_weights.eval()

    model_zonder_adaptive_weights = generate_model()
    model_zonder_adaptive_weights.load_state_dict(torch.load(MODEL_ZONDER_PAD, map_location='cpu'))
    model_zonder_adaptive_weights.eval()

    return model_adaptive_weights, model_zonder_adaptive_weights


def bereken_voorspellingen(model_adaptive_weights: nn.Sequential, model_zonder_adaptive_weights: nn.Sequential) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bereken de voorspellingen van beide modellen op hetzelfde rooster.

    Args:
        model_adaptive_weights (nn.Sequential): Het getraine model met dynamische gewichten
        model_zonder_adaptive_weights (nn.Sequential): Het getraine model zonder dynamische gewichten

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Een tuple met de voorspellingen van beide modellen en het absolute verschil
    """
    x_waarden = np.linspace(0, L, AANTAL_X)
    u_mat_imp = np.zeros((AANTAL_T + 1, AANTAL_X))
    nn_mat = np.zeros((AANTAL_T + 1, AANTAL_X))
    t_waarden = np.linspace(0, T, AANTAL_T + 1)

    # Zet de x-coördinaten één keer om; alleen de tijdstap verandert per iteratie.
    x_tensor = torch.tensor(x_waarden, dtype=torch.float32).view(-1, 1)

    print("Start voorspellingen Neuraal Netwerk...")
    with torch.no_grad():
        for n, t_val in enumerate(t_waarden):
            t_tensor = torch.full_like(x_tensor, t_val)
            test_punten = torch.cat((x_tensor, t_tensor), dim=1)

            u_pred_aw = model_adaptive_weights(test_punten)
            nn_mat[n, :] = u_pred_aw.view(-1).cpu().numpy()

            u_pred_zonder_aw = model_zonder_adaptive_weights(test_punten)
            u_mat_imp[n, :] = u_pred_zonder_aw.view(-1).cpu().numpy()

    print("Voorspellingen Neuraal Netwerk voltooid!")
    verschil = np.abs(nn_mat - u_mat_imp)
    l2_rel = np.linalg.norm(nn_mat - u_mat_imp) / np.linalg.norm(u_mat_imp)
    print(f"Relatieve L2-fout tussen de methoden: {l2_rel}")

    return nn_mat, u_mat_imp, verschil


def plot_heatmaps(nn_mat: np.ndarray, verschil: np.ndarray, u_mat_imp: np.ndarray):
    """Toon de heatmaps voor dynamische gewichten, fout en zonder dynamische gewichten.

    Args:
        nn_mat (np.ndarray): Voorspelling van het neuraal netwerk
        verschil (np.ndarray): Het absoluut verschil tussen het neuraal netwerk en de semi-impliciete methode
        u_mat_imp (np.ndarray): Voorspelling van de semi-impliciete methode (referentie-oplossing)
    """
    plt.figure(figsize=(14, 5), num=2)

    plt.subplot(1, 3, 1)
    plt.imshow(nn_mat, extent=(0, L, 0, T), aspect='auto', origin='lower', cmap='viridis')
    plt.title("Met Dynamische Gewichten", fontsize=14)
    plt.ylabel("Tijd (t)", fontsize=14)
    plt.xlabel("Positie (x)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    cbar = plt.colorbar()
    cbar.set_label('Oplossing u(x,t)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.subplot(1, 3, 2)
    plt.imshow(verschil, extent=(0, L, 0, T), aspect='auto', origin='lower', cmap='viridis')
    plt.title("Absoluut Verschil", fontsize=14)
    plt.ylabel("Tijd (t)", fontsize=14)
    plt.xlabel("Positie (x)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    cbar2 = plt.colorbar()
    cbar2.set_label('Absoluut Verschil', fontsize=14)
    cbar2.ax.tick_params(labelsize=12)

    plt.subplot(1, 3, 3)
    plt.imshow(u_mat_imp, extent=(0, L, 0, T), aspect='auto', origin='lower', cmap='viridis')
    plt.title("Zonder Dynamische Gewichten", fontsize=14)
    plt.ylabel("Tijd (t)", fontsize=14)
    plt.xlabel("Positie (x)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    cbar3 = plt.colorbar()
    cbar3.set_label('Oplossing u(x,t)', fontsize=14)
    cbar3.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(HEATMAP_PAD, dpi=300)
    plt.show()


def main():
    """Voer de volledige workflow uit voor de vergelijking met en zonder dynamische gewichten."""
    waarden = laad_leersnelheid_gegevens()

    (
        _, k_infinity_adaptive, alpha_adaptive, model_adaptive,
        _, k_infinity, alpha, model_zonder,
        _, k_infinity_slecht, alpha_slecht, model_slecht,
    ) = waarden

    k_infinity_adaptive = naar_float(k_infinity_adaptive)
    alpha_adaptive = naar_float(alpha_adaptive)
    k_infinity = naar_float(k_infinity)
    alpha = naar_float(alpha)
    k_infinity_slecht = naar_float(k_infinity_slecht)
    alpha_slecht = naar_float(alpha_slecht)

    t = np.linspace(0, 15, 50)
    k_adaptive = leer_curve(t, k_infinity_adaptive, alpha_adaptive)
    k_zonder_adaptive = leer_curve(t, k_infinity, alpha)
    k_slecht = leer_curve(t, k_infinity_slecht, alpha_slecht)

    plot_leersnelheid(t, k_adaptive, k_zonder_adaptive, k_slecht, alpha_adaptive, alpha, alpha_slecht)

    if model_adaptive is None:
        raise RuntimeError("PINN did not return the adaptive model. Add 'model' to returns in the PINN call.")
    if model_zonder is None:
        raise RuntimeError("PINN did not return the non-adaptive model. Add 'model' to returns in the PINN call.")
    model_adaptive_weights = model_adaptive.to('cpu').eval()
    model_zonder_adaptive_weights = model_zonder.to('cpu').eval()
    nn_mat, u_mat_imp, verschil = bereken_voorspellingen(model_adaptive_weights, model_zonder_adaptive_weights)
    plot_heatmaps(nn_mat, verschil, u_mat_imp)


if __name__ == '__main__':
    main()