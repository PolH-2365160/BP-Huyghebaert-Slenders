import sys
import os
import numpy as np
from torch import nn
import torch
import matplotlib.pyplot as plt
import torch.optim as optim

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../")) 

if root_dir not in sys.path:
    sys.path.append(root_dir)


# 1. Definieer een functie om het model te creëren met een specifieke architectuur.
def create_model()-> nn.Sequential:
    """Creeer een neuraal netwerk model met een specifieke architectuur.
    Returns:
        nn.Sequential: Het gedefinieerde neuraal netwerk model.
    """
    model = nn.Sequential(
        nn.Linear(1, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )
    return model

# 2. Definieer de verliesfunctie die zowel de PDE-residuen als de data-gedreven componenten omvat, afhankelijk van het type model.
def loss_function(output: torch.Tensor, model: nn.Sequential, Tenv: float, t: torch.Tensor, t_data: torch.Tensor, T0: float, data: torch.Tensor, soort: str, r: float) -> torch.Tensor:
    """Bereken de totale verliesfunctie voor het trainen van het model, afhankelijk van het type model (data-gedreven, UPINN of PINN).
    Args:        
        output (torch.Tensor): De voorspellingen van het model.
        model (nn.Sequential): Het neuraal netwerk model.
        Tenv (float): De omgevingstemperatuur.
        t (torch.Tensor): De tijdstensor voor het berekenen van de PDE-residuen.
        t_data (torch.Tensor): De tijdstensor voor de data-gedreven component.
        T0 (float): De initiële temperatuur.
        data (torch.Tensor): De gemeten data voor de data-gedreven component.
        soort (str): Het type model ('data', 'UPINN' of 'PINN').
        r (float): De koelingsconstante.
    Returns:
        torch.Tensor: De totale verlieswaarde voor het trainen van het model.
    """
    T = output
    T_t = torch.autograd.grad(
        T, t, grad_outputs=torch.ones_like(T), create_graph=True
    )[0]

    # Bereken de PDE-residuen
    pde_residual = T_t - r*(Tenv - T)
    loss_pde = torch.mean(pde_residual**2)
    
    # Bereken de initiële conditie verlies
    initial_condition = T0
    loss_ic = torch.mean((initial_condition-T[0])**2)
    if soort == 'data-gedreven':
        loss_pde = torch.zeros_like(loss_pde)
        loss_ic = torch.zeros_like(loss_ic)


    # Bereken de data-gedreven component van het verlies
    predictions = model(t_data)
    data_loss = torch.mean((predictions-data)**2)
    if soort == 'UPINN':
        data_loss = torch.zeros_like(data_loss)

    # Combineer de totale loss met een geschikte weging
    lam_pde = 70
    if soort == 'UPINN':   
        total_loss = loss_ic + lam_pde * loss_pde 
        return total_loss
    if soort == 'data-gedreven':   
        total_loss = data_loss
        return total_loss
    else:  
        total_loss = loss_ic + lam_pde * loss_pde + data_loss
        return total_loss
   
def modellen_trainen(soort: str, T0: float, Tenv: float, r: float, data: torch.Tensor, t_data: torch.Tensor) -> tuple[nn.Sequential, torch.Tensor, torch.Tensor]:
    """Train een neuraal netwerk model op basis van het opgegeven type (data-gedreven, UPINN or PINN) en retourneer het getrainde model samen met de data en tijdstensor.

    Args:
        soort (str): Het type model ('data-gedreven', 'UPINN' or 'PINN').
        T0 (float): De initiële temperatuur.
        Tenv (float): De omgevingstemperatuur.
        r (float): De koelingsconstante.
        data (torch.Tensor): De gemeten data voor de data-gedreven component.
        t_data (torch.Tensor): De tijdstensor voor de data-gedreven component.

    Returns:
        tuple[nn.Sequential, torch.Tensor, torch.Tensor]: Het getrainde model samen met de data en tijdstensor.
    """

    # 1. Creëer het model
    gridsize = 5000
    t = torch.linspace(0, 1000, gridsize, requires_grad=True).reshape(-1,1)
    model = create_model()
    optimizer = optim.Adam(list(model.parameters()), lr=0.001)
    model.train()

    # 2. Train het model
    EPOCHS = 8000
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        output = model(t)
        loss = loss_function(output, model, Tenv, t, t_data, T0, data, soort,r)
        loss.backward(retain_graph=True)
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch} van {soort} model, Loss: {loss.item()}")

    print(f"{soort} model is opgeslagen!")
    return model, data, t_data

# 3. Definieer de parameters
index = 0
r = 0.005
Tenv = 25
T0 = 100
L = 10
T = 1000
r = 0.005
D = 1 
aantal_x = 10000
aantal_t = 3200
t_waarden = np.linspace(0, T, aantal_t+1)
del_x = L / (aantal_x-1)
del_t = T / aantal_t
Nt = 2000

# 4. Genereer de data met ruis
t_data = torch.linspace(0,300,10,dtype=torch.float32).reshape(-1,1)
data = Tenv + (T0 - Tenv)*torch.exp(-r*t_data)
noise = torch.randn_like(data)
data += noise
data.reshape(-1,1)

# 5. Train de modellen en verkrijg de voorspellingen
model_data, data_data, t_data = modellen_trainen('data-gedreven', T0, Tenv, r, data, t_data)
model_comb, data_comb, t_data_comb = modellen_trainen('PINN', T0, Tenv, r, data, t_data)
model_pinn , data_pinn, t_data_pinn= modellen_trainen('UPINN', T0, Tenv, r, data, t_data)
print("Alle modellen zijn getraind en klaar voor vergelijking.")
model_data.eval()
model_comb.eval()
model_pinn.eval()

# 6. Bereken de exacte oplossing
exacte_oplossing = (100-25)*np.exp(-0.005*t_waarden) + 25

# 7. Zet de tijdstensor om naar een geschikte vorm voor voorspelling en maak voorspellingen zonder gradients te berekenen
t_eval = torch.tensor(t_waarden.reshape(-1, 1), dtype=torch.float32)

with torch.no_grad():
    u_pred_data = model_data(t_eval)
    u_pred_comb = model_comb(t_eval)
    u_pred_pinn = model_pinn(t_eval)

# 8. Data omzetten naar NumPy voor Matplotlib
t_plot = t_eval.numpy().flatten()
u_plot_data = u_pred_data.numpy().flatten()
u_plot_comb = u_pred_comb.numpy().flatten()
u_plot_pinn = u_pred_pinn.numpy().flatten()


# 9. Plot de resultaten
plt.figure(figsize=(10, 6))
plt.plot(t_waarden, exacte_oplossing, label='Exacte oplossing', color='red', linewidth=2, linestyle='--')
plt.plot(t_plot, u_plot_data, label='Data-Driven Neuraal netwerk', color='blue')
plt.plot(t_plot, u_plot_comb, label='PINN', color='green')
plt.plot(t_plot, u_plot_pinn, label='UPINN', color='orange')
plt.axvline(x = 300, color = 'black', alpha = 0.4, label = 'Grens trainingsdata')
plt.xlabel('Tijd (s)', fontsize=14)
plt.ylabel('Temperatuur ($^\\circ C$)', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Vergelijking Neurale Netwerken', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(current_dir + '/Vergelijking_neurale_netwerken.png', dpi=300)
plt.show()
