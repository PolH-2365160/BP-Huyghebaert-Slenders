import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Setup
current_dir = Path(__file__).parent
torch.manual_seed(0)
np.random.seed(0)

# Constants
GRIDSIZE = 5000
T_END = 85
EPOCHS = 8000
LR = 0.001
LAM_PDE_PINN = 70
LAM_PDE_UPINN = 1

# Collocation points
t = torch.linspace(0, T_END, GRIDSIZE, requires_grad=True).reshape(-1, 1)


def generate_model()->nn.Sequential:
    """Genereer een neuraal netwerk met 3 hidden layers van 64 neuronen.

    Returns:
        nn.Sequential: Het gegenereerde neuraal netwerk
    """
    return nn.Sequential(
        nn.Linear(1, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )


def loss_function_pinn(output: torch.Tensor, model: nn.Module, t_data: torch.Tensor, data_v: torch.Tensor, t_collocation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Loss functie voor PINN model (PDE: v' = 1).

    Args:
        output (torch.Tensor): Voorspelligen van het model op de collocation punten
        model (nn.Module): Het gegenereerde neuraal netwerk
        t_data (torch.Tensor): Data punten waarop de data loss wordt berekend
        data_v (torch.Tensor): De exacte waarden van v op de data punten
        t_collocation (torch.Tensor): Collocation punten waarop de PDE-residuals worden berekend

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Totale loss, PDE loss, Initial Condition loss, Data loss
    """
    
    v = output

    v_t = torch.autograd.grad(
        v, t_collocation, grad_outputs=torch.ones_like(v), create_graph=True
    )[0]
    pde_residual = v_t - 1.0
    loss_pde = torch.mean(pde_residual**2)

    loss_ic = torch.mean((0.0 - v[0])**2)

    predictions_v = model(t_data)
    loss_data = torch.mean((predictions_v - data_v)**2)

    total_loss = loss_ic + LAM_PDE_PINN * loss_pde + loss_data
    return total_loss, loss_pde, loss_ic, loss_data


def loss_function_upinn(output: torch.Tensor, t_collocation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Loss functie voor UPINN model (PDE: u' = u).

    Args:
        output (torch.Tensor): Voorspelligen van het model op de collocation punten
        t_collocation (torch.Tensor): Collocation punten waarop de PDE-residuals worden berekend

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Totale loss, PDE loss, Initial Condition loss
    """
    u = output

    u_t = torch.autograd.grad(
        u, t_collocation, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    pde_residual = u_t - u
    loss_pde = torch.mean(pde_residual**2)

    loss_ic = torch.mean((1.0 - u[0])**2)

    total_loss = loss_ic + LAM_PDE_UPINN * loss_pde
    return total_loss, loss_pde, loss_ic


def train_pinn(t_collocation: torch.Tensor, epochs: int=EPOCHS)->nn.Module:
    """Traint een gecombineerd PINN model.

    Args:
        t_collocation (torch.Tensor): Collocation punten waarop de PDE-residuals worden berekend
        epochs (int, optional): Aantal trainingsiteraties om te trainen. Defaults to EPOCHS.

    Returns:
        nn.Module: Het getrainde model
    """
    t_data = torch.linspace(0, T_END, 3, dtype=torch.float32).reshape(-1, 1)
    data_u = torch.exp(t_data)
    data_v = torch.log(data_u)

    model = generate_model()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()

    print("Training combinatie model...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(t_collocation)
        loss, loss_pde, loss_ic, loss_data = loss_function_pinn(
            output, model, t_data, data_v, t_collocation
        )
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:4d}/{epochs}: Loss = {loss.item():.2e}")

    return model


def train_upinn(t_collocation: torch.Tensor, epochs: int=EPOCHS)->nn.Module:
    """Traint een UPINN model.

    Args:
        t_collocation (torch.Tensor): Collocation punten waarop de PDE-residuals worden berekend
        epochs (int, optional): Aantal trainingsiteraties om te trainen. Defaults to EPOCHS.

    Returns:
        nn.Module: Het getrainde model
    """
    model = generate_model()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()

    print("Training UPINN model...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(t_collocation)
        loss, loss_pde, loss_ic = loss_function_upinn(output, t_collocation)
        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 500 == 0:
            print(f"  Epoch {epoch:4d}/{epochs}: Loss = {loss.item():.2e}")

    return model


def plot_pinn_results(model: nn.Module, t_plot_nn: torch.Tensor)->None:
    """Plot resultaten van combinatie model.

    Args:
        model (nn.Module): Een getraind neuraal netwerk 
        t_plot_nn (torch.Tensor): Collocation punten waarop de voorspellingen worden gemaakt
    """
    t_plot_exact = np.linspace(0, T_END, 500, dtype=np.float64).reshape(-1, 1)
    exacte_oplossing = np.exp(t_plot_exact).flatten()

    v_pred = model(t_plot_nn).detach().numpy()
    u_pred = np.exp(v_pred.astype(np.float64)).flatten()

    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(t_plot_exact, exacte_oplossing, label="Exacte oplossing",
             color='tab:red', linewidth=5, alpha=0.7, linestyle='dashed')
    plt.plot(t_plot_nn, u_pred, label='Neuraal netwerk',
             color='black', linewidth=1.5, linestyle='-')

    plt.yscale('symlog', linthresh=1e-1)
    plt.ylim(-1, 1e45)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Exponentiële Groei via PINN", fontsize=16)
    plt.xlabel("Tijd (t)", fontsize=14)
    plt.ylabel("Waarde (u)", fontsize=14)
    plt.legend(fontsize=14)

    plt.savefig(current_dir / 'luiheid_voorbeeld_pinn.png', dpi=300, bbox_inches='tight')


def plot_upinn_results(model: nn.Module, t_plot_nn: torch.Tensor)->None:
    """Plot resultaten van UPINN model.

    Args:
        model (nn.Module): Een getraind neuraal netwerk 
        t_plot_nn (torch.Tensor): Collocation punten waarop de voorspellingen worden gemaakt
    """
    t_plot_exact = np.linspace(0, T_END, 500, dtype=np.float64).reshape(-1, 1)
    exacte_oplossing = np.exp(t_plot_exact).flatten()

    u_pred = model(t_plot_nn).detach().numpy()

    plt.figure(figsize=(8, 6))
    plt.grid(True)
    plt.hlines(0.0, 0, T_END, label='Nul-oplossing',
               color='tab:orange', linewidth=5, alpha=0.7, linestyle='dashed')
    plt.plot(t_plot_exact, exacte_oplossing, label="Exacte oplossing",
             color='tab:red', linewidth=5, alpha=0.7, linestyle='dashed')
    plt.plot(t_plot_nn, u_pred, label='Neuraal netwerk',
             color='black', linewidth=1.5, linestyle='-')

    plt.yscale('symlog', linthresh=1e-1)
    plt.ylim(-1, 1e40)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Exponentiële Groei via UPINN", fontsize=16)
    plt.xlabel("Tijd (t)", fontsize=14)
    plt.ylabel("Waarde (u)", fontsize=14)
    plt.legend(fontsize=14, loc='upper left')

    plt.savefig(current_dir / 'luiheid_voorbeeld_upinn.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    model_pinn= train_pinn(t, EPOCHS)
    model_upinn = train_upinn(t, EPOCHS)

    t_plot_nn = torch.linspace(0, T_END, 500, dtype=torch.float32).reshape(-1, 1)

    plot_pinn_results(model_pinn, t_plot_nn)
    plot_upinn_results(model_upinn, t_plot_nn)

    plt.show()