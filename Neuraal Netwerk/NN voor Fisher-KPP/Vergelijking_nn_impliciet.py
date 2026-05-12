import numpy as np
import matplotlib.pyplot as plt
import sys
from torch import nn
import torch
from pathlib import Path
import numpy.typing as npt

# Maak het projectroot-pad beschikbaar voor imports.
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from Functies.Impliciete_functie import impliciete_oplossing



# Model- en simulatieparameters.
L = 20.0
T = 20.0
r = 1.0
D = 1
aantal_x = 1000
aantal_t = 32000


def maak_model()->nn.Sequential:
    """Bouwt de neurale netwerktopologie die voor de vergelijking gebruikt wordt.

    Returns:
        nn.Sequential: Een ongetraind neuraal netwerk met 5 hidden layers van 50 neuronen elk.
    """
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


def laad_getraind_model():
    """Laadt het eerder getrainde model uit het bestandssysteem."""
    model = maak_model()
    model_path = root_dir / 'Neuraal Netwerk' / 'Getrainde_modellen' / 'model_adaptive_weights_exp_100000_epoches_80000_data.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def bereken_impliciete_oplossing()->tuple[np.ndarray, np.ndarray]:
    """Bereken de impliciete referentie-oplossing voor alle tijdstappen.

    Returns:
        tuple[np.ndarray, np.ndarray]: De x-waarden en de matrix met de impliciete oplossingen.
    """
    x_waarden = np.linspace(0, L, aantal_x)
    u0 = 1 / (1 + np.exp(x_waarden - 10))

    u_mat_imp = np.zeros((aantal_t + 1, aantal_x))
    u_imp = u0.copy()
    u_mat_imp[0] = u_imp

    for n in range(aantal_t):
        u_imp = impliciete_oplossing(L, T, r, D, aantal_x, aantal_t, u_imp)
        u_mat_imp[n + 1] = u_imp

    return x_waarden, u_mat_imp


def bereken_nn_voorspellingen(model: nn.Sequential, x_waarden: np.ndarray) -> np.ndarray:
    """Bereken de voorspellingen van het neurale netwerk voor alle tijdstappen.

    Args:
        model (nn.Sequential): Het getrainde neurale netwerk.
        x_waarden (np.ndarray): De x-waarden voor het rooster.

    Returns:
        np.ndarray: De matrix met de voorspellingen van het neurale netwerk.
    """
    print("Start voorspellingen Neuraal Netwerk...")
    nn_mat = np.zeros((aantal_t + 1, aantal_x))
    t_waarden = np.linspace(0, T, aantal_t + 1)

    # Zet x één keer om naar een tensor; alleen de tijd verandert per iteratie.
    x_tensor = torch.tensor(x_waarden, dtype=torch.float32).view(-1, 1)

    with torch.no_grad():
        for n, t_val in enumerate(t_waarden):
            t_tensor = torch.full_like(x_tensor, t_val)
            test_punten = torch.cat((x_tensor, t_tensor), dim=1)
            u_pred = model(test_punten)
            nn_mat[n, :] = u_pred.view(-1).numpy()

    return nn_mat


def bereken_fouten(nn_mat: npt.NDArray[np.float32], u_mat_imp: npt.NDArray[np.float32]) -> tuple[npt.NDArray, npt.Float32, npt.Float32]:
    """Bereken absolute foutmaten tussen het netwerk en de impliciete oplossing.

    Args:
        nn_mat (np.ndarray): Voorspelling van neuraal netwerk
        u_mat_imp (np.ndarray): Voorspelling van impliciete oplossing

    Returns:
        tuple[np.ndarray, float, float]: Absoluut verschil, absolute L^2 norm, relatieve L^2 norm
    """
    verschil = np.abs(nn_mat - u_mat_imp)
    l2_abs = np.linalg.norm(nn_mat - u_mat_imp)
    l2_rel = l2_abs / np.linalg.norm(u_mat_imp)
    return verschil, l2_abs, l2_rel


def plot_resultaten(nn_mat: np.ndarray, verschil: np.ndarray, u_mat_imp: np.ndarray) -> None:
    """Toon de drie heatmaps naast elkaar voor een directe vergelijking.

    Args:
        nn_mat (np.ndarray): Voorspelling neuraal netwerk
        verschil (np.ndarray): Absoluut verschil tussen neuraal netwerk en impliciete oplossing
        u_mat_imp (np.ndarray): Impliciete referentie-oplossing
    """
    plt.figure(figsize=(14, 5))

    # Linker plot: Neuraal netwerk.
    plt.subplot(1, 3, 1)
    plt.imshow(nn_mat, extent=(0, L, 0, T), aspect='auto', origin='lower', cmap='viridis')
    plt.title("Neuraal Netwerk", fontsize=14)
    plt.ylabel("Tijd (t)", fontsize=14)
    plt.xlabel("Positie (x)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    cbar = plt.colorbar()
    cbar.set_label('Oplossing u(x,t)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Midden plot: absolute fout.
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

    # Rechter plot: impliciete referentie-oplossing.
    plt.subplot(1, 3, 3)
    plt.imshow(u_mat_imp, extent=(0, L, 0, T), aspect='auto', origin='lower', cmap='viridis')
    plt.title("Semi-impliciete Oplossing", fontsize=14)
    plt.ylabel("Tijd (t)", fontsize=14)
    plt.xlabel("Positie (x)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    cbar3 = plt.colorbar()
    cbar3.set_label('Oplossing u(x,t)', fontsize=14)
    cbar3.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(current_dir / 'vergelijking_NN_impliciet.png', dpi=300)
    plt.show()


def main():
    """Voert de volledige vergelijking uit en plot de resultaten."""
    model = laad_getraind_model()
    x_waarden, u_mat_imp = bereken_impliciete_oplossing()
    nn_mat = bereken_nn_voorspellingen(model, x_waarden)
    verschil, l2_abs, l2_rel = bereken_fouten(nn_mat, u_mat_imp)

    print(f"Absolute L2-fout: {l2_abs:.6f}")
    print(f"Relatieve L2-fout: {l2_rel:.6f}")

    plot_resultaten(nn_mat, verschil, u_mat_imp)


if __name__ == "__main__":
    main()