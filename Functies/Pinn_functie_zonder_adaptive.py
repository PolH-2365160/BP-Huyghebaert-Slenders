import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
from scipy.optimize import curve_fit
try:
    from Functies.Impliciete_functie import impliciete_oplossing
except ImportError:
    from Impliciete_functie import impliciete_oplossing


# --- DEFINITIE LOSS FUNCTIE ---    
def compute_losses(model: nn.Sequential, x_pde: torch.Tensor, t_pde: torch.Tensor, x_data: torch.Tensor, t_data: torch.Tensor, target_data: torch.Tensor, r: float, D: float, L: float,T: float, N_bdy: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Loss functie die de trainingsloop gebruikt om het model te trainen

    Args:
        model (nn.Sequential): Het neuraal netwerk model dat getraind wordt
        x_pde (torch.Tensor): Tensor met de x-coördinaten van de willekeurige punten voor de PDE loss
        t_pde (torch.Tensor): Tensor met de t-coördinaten van de willekeurige punten voor de PDE loss
        x_data (torch.Tensor): Tensor met de x-coördinaten van de data punten
        t_data (torch.Tensor): Tensor met de t-coördinaten van de data punten
        target_data (torch.Tensor): Tensor met de doelwaarden voor de data punten
        r (float): De groeisnelheid van de populatie
        D (float): De diffusiecoëfficiënt
        L (float): De lengte van het domein
        T (float): De tijdsduur van de simulatie
        N_bdy (int): Het aantal boundary points dat wordt gebruikt voor de boundary loss
        device (torch.device): Het apparaat waarop de tensors zich bevinden (CPU, CUDA, MPS)
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Een tuple met de vier componenten van de loss: (loss_pde, loss_init, loss_bdy, loss_data)
    """
    # 1. PDE Loss
    u_pde = model(torch.cat((x_pde, t_pde), 1))
    u_t = torch.autograd.grad(u_pde, t_pde, grad_outputs=torch.ones_like(u_pde), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pde, x_pde, grad_outputs=torch.ones_like(u_pde), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_pde, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    pde_residual = u_t - D * u_xx - r * u_pde * (1 - u_pde)
    loss_pde = torch.mean(pde_residual**2)
    
    # 2. Beginvoorwaarde (t=0)
    x_0 = torch.empty(N_bdy, 1, device=device).uniform_(0, L)
    t_0 = torch.zeros_like(x_0)
    u_0_pred = model(torch.cat((x_0, t_0), 1))
    initial_condition = 1 / (1 + torch.exp(x_0 - 10))
    # initial_condition = 0.5 * (1 - torch.tanh(x_0 / 2))
    # initial_condition = 0.8*torch.exp(-(x_0 - 5)**2)+0.8*torch.exp(-(x_0 - 15)**2)
    # initial_condition = 0.1*(torch.sin(torch.pi * x_0 / 5))**2
    loss_init = torch.mean((u_0_pred - initial_condition)**2)

    # 3. Randvoorwaarden (x=0 en x=L)
    t_bdy = torch.empty(N_bdy, 1, device=device).uniform_(0, T)
    x_bdy_0 = torch.zeros_like(t_bdy).requires_grad_(True)
    u_bdy_0 = model(torch.cat((x_bdy_0, t_bdy), 1))
    u_x_0 = torch.autograd.grad(u_bdy_0, x_bdy_0, grad_outputs=torch.ones_like(u_bdy_0), create_graph=True)[0]
    loss_bdy1 = torch.mean(u_x_0**2) 
    x_bdy_L = torch.full_like(t_bdy, L).requires_grad_(True)
    u_bdy_L = model(torch.cat((x_bdy_L, t_bdy), 1))
    u_x_L = torch.autograd.grad(u_bdy_L, x_bdy_L, grad_outputs=torch.ones_like(u_bdy_L), create_graph=True)[0]
    loss_bdy2 = torch.mean(u_x_L**2)
    loss_bdy = loss_bdy1 + loss_bdy2

    # 4. Data loss
    predictions = model(torch.cat((x_data, t_data), dim=1))
    loss_data = torch.mean((predictions - target_data)**2)
    
    return loss_pde, loss_init, loss_bdy, loss_data

def Pinn_zonder_adaptive_weights(basis_pad: Path, seed: int = 42, naam_model: str = '', L: int = 20, T: int = 15, r: int = 1, D: int = 1, EPOCHS: int = 8000, Nx: int = 20000, Nt: int = 3200, N_colloc: int = 5000, N_bdy: int = 1000, 
                                    aantal_trainingspunten: int = 5000, lam_pde: float = 1.0, lam_init: float = 1.0, lam_bdy: float = 1.0, lam_data: float = 1.0, plot_total_loss_name: str = '', plot_comparison_loss_name: str = '',
                                    returns: list[str] = [], ruis: float = 0):
    """Functie om een PINN te trainen zonder adaptieve gewichten, waarbij de gewichten van de verschillende componenten van de loss constant blijven gedurende de training.

    Args:
        basis_pad (Path): Het pad voor de basisfuncties
        seed (int, optional): De seed voor de random generator. Defaults to 42.
        naam_model (str, optional): De naam van het model. Defaults to ''.
        L (int, optional): De lengte van het domein. Defaults to 20.
        T (int, optional): De tijdslimiet. Defaults to 15.
        r (int, optional): Het ratio parameter. Defaults to 1.
        D (int, optional): Het diffusiecoefficient. Defaults to 1.
        EPOCHS (int, optional): Het aantal epochs. Defaults to 8000.
        Nx (int, optional): Het aantal x-punten. Defaults to 20000.
        Nt (int, optional): Het aantal t-punten. Defaults to 3200.
        N_colloc (int, optional): Het aantal collocation points. Defaults to 5000.
        N_bdy (int, optional): Het aantal boundary points. Defaults to 1000.
        aantal_trainingspunten (int, optional): Het aantal trainingspunten. Defaults to 5000.
        lam_pde (float, optional): Het gewicht voor de PDE loss. Defaults to 1.0.
        lam_init (float, optional): Het gewicht voor de initiële conditie loss. Defaults to 1.0.
        lam_bdy (float, optional): Het gewicht voor de boundary conditie loss. Defaults to 1.0.
        lam_data (float, optional): Het gewicht voor de data loss. Defaults to 1.0.
        plot_total_loss_name (str, optional): De naam voor de total loss plot. Defaults to ''.
        plot_comparison_loss_name (str, optional): De naam voor de comparison loss plot. Defaults to ''.
        returns (list[str], optional): De waarden die worden geretourneerd. Defaults to [].
        ruis (float, optional): Het niveau van ruis in de data. Defaults to 0.

    Raises:
        ValueError: Te veel trainingspunten geselecteerd, meer dan het totale aantal beschikbare punten in de data-generatie.
        
    Returns:
        _type_: Afhankelijk van de inhoud van de 'returns' parameter, kunnen verschillende waarden worden geretourneerd. Mogelijke waarden zijn onder andere: 'Leersnelheid' (de geschiedenis van K_t waarden), 'losses_history' (de geschiedenis van de totale loss en de individuele componenten van de loss), en andere afhankelijk van de implementatie.
    """
    
    # --- SETUP ---
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")


    # --- DATA GENEREREN ---
    x = torch.linspace(0, L, Nx, requires_grad=True, device=device)
    t = torch.linspace(0, T, Nt, requires_grad=True, device=device)
    x_grid, t_grid = torch.meshgrid(x, t, indexing="ij") # Zorg ervoor dat x varieert langs de eerste dimensie en t langs de tweede
    x = x_grid.T.reshape(-1, 1) # Transponeer voordat je reshape toepast, zodat x langs de eerste dimensie loopt
    t = t_grid.T.reshape(-1, 1) # Transponeer voordat je reshape toepast, zodat t langs de eerste dimensie loopt

    t_punten = torch.linspace(0, T, Nt, device=device) # Voor de data-generatie gebruiken we dezelfde t-punten als in het grid
    x_punten = torch.linspace(0, L, 25, device=device) # We gebruiken een kleiner aantal x-punten voor de data-generatie, omdat we later willekeurige trainingspunten zullen selecteren
    x_test_grid, t_test_grid = torch.meshgrid(x_punten, t_punten, indexing="ij") # Maak een grid van x en t voor de testdata
    x_test_data = x_test_grid.T.reshape(-1, 1) 
    t_test_data = t_test_grid.T.reshape(-1, 1)
    Nx_data = len(x_punten) # Aantal x-punten in de data-generatie
    Nt_data = len(t_punten) # Aantal t-punten in de data-generatie

    # Beginvoorwaarde kiezen en data-array initialiseren
    u0 = 1 / (1 + torch.exp(x_punten - 10)).cpu().numpy()
    # u0= 0.5 * (1 - torch.tanh(x_punten / 2)).cpu().numpy()
    # u0 = (0.8*torch.exp(-(x_punten - 5)**2)+0.8*torch.exp(-(x_punten - 15)**2)).cpu().numpy()
    # u0 = (0.1*(torch.sin(torch.pi * x_punten / 5))**2).cpu().numpy()
    data_array = np.zeros((Nx_data, Nt_data))
    data_array[:, 0] = u0
    del_t_data = T / (Nt_data - 1)

    # Bereken de data door iteratief de impliciete oplossing toe te passen op de vorige tijdstap
    u_prev = u0.copy()
    for n in range(1, Nt_data):
        u_next = impliciete_oplossing(
            L=L, T=del_t_data, r=r, D=D, aantal_x=Nx_data, aantal_t=1, u=u_prev
        )
        data_array[:, n] = u_next
        u_prev = u_next

    # Converteer de data-array naar een PyTorch tensor en reshape deze zodat elke rij overeenkomt met een (x, t) paar
    data = torch.tensor(data_array.T.reshape(-1, 1), device=device, dtype=torch.float32)

    # Voeg optioneel ruis toe aan de data
    if ruis != 0:
        std_data = torch.std(data)
        ruis_tensor = ruis * std_data * torch.randn_like(data)
        data += ruis_tensor
    
    if aantal_trainingspunten > x_test_data.shape[0]:
        raise ValueError(f"Aantal trainingspunten ({aantal_trainingspunten}) is groter dan het totale aantal beschikbare punten ({x_test_data.shape[0]}). Verlaag het aantal trainingspunten.")

    # Willekeurige indices genereren voor het selecteren van trainings- en validatiepunten
    totaal_punten = x_test_data.shape[0]
    willekeurige_indices = torch.randperm(totaal_punten)

    # Selecteer de trainingspunten op basis van de willekeurige indices
    train_indices = willekeurige_indices[:aantal_trainingspunten]
    x_train_data = x_test_data[train_indices]
    t_train_data = t_test_data[train_indices]
    trainings_data = data[train_indices]

    # De resterende punten worden gebruikt voor validatie
    val_indices = willekeurige_indices[aantal_trainingspunten:]
    x_val_data = x_test_data[val_indices]
    t_val_data = t_test_data[val_indices]
    val_data = data[val_indices]


    # --- MODEL, OPTIMIZER EN LOSS FUNCTIE DEFINIEREN ---
    model = nn.Sequential(
        nn.Linear(2, 50), nn.Tanh(),
        nn.Linear(50, 50), nn.Tanh(),
        nn.Linear(50, 50), nn.Tanh(),
        nn.Linear(50, 50), nn.Tanh(),
        nn.Linear(50, 50), nn.Tanh(),
        nn.Linear(50, 1),
    ).to(device)

    optimizer = optim.Adam(list(model.parameters()), lr=0.001)
    model.train()

    # --- TRAINING LOOP ZONDER ADAPTIEVE GEWICHTEN ---

    # 1. Lijsten om de geschiedenis van de losses en K_t op te slaan
    loss_hist_tot, loss_hist_pde, loss_hist_init, loss_hist_bdy, loss_hist_data = [], [], [], [], []
    K_history = []

    # 2. Training loop
    start_time = time.time()
    last_100_time = start_time
    for epoch in range(EPOCHS):
        # Reset de gradients van het model
        optimizer.zero_grad()
        
        # Genereer willekeurige collocatiepunten voor de PDE loss
        x_batch = torch.empty(N_colloc, 1, device=device).uniform_(0, L).requires_grad_(True)
        t_batch = torch.empty(N_colloc, 1, device=device).uniform_(0, T).requires_grad_(True)
        
        # Bereken de vier componenten van de loss met behulp van de compute_losses functie
        loss_pde, loss_init, loss_bdy, loss_data = compute_losses(
            model, x_batch, t_batch, x_train_data, t_train_data, trainings_data,r ,D ,L ,T ,N_bdy , device
        )
        
        # Bereken de totale loss als een gewogen som van de vier componenten
        total_loss = (lam_pde * loss_pde) + (lam_init * loss_init) + (lam_bdy * loss_bdy) + (lam_data * loss_data)
        
        # Backpropagation en optimalisatie stap
        total_loss.backward()
        optimizer.step()

        # Optioneel: Bereken K_t op basis van de validatie MSE en sla deze op in K_history
        if 'Leersnelheid' in returns:
            with torch.no_grad():
                val_predictions = model(torch.cat((x_val_data, t_val_data), dim=1))
                val_mse = torch.mean((val_predictions - val_data)**2).item()
                K_t = 1/(1+ val_mse)  # Voorbeeld van een K_t functie gebaseerd op de validatie MSE
                K_history.append(K_t)
        
        # Sla de huidige totale loss en de individuele componenten van de loss op in de geschiedenis voor latere visualisatie
        loss_hist_tot.append(total_loss.item())
        loss_hist_pde.append(loss_pde.item())
        loss_hist_init.append(loss_init.item())
        loss_hist_bdy.append(loss_bdy.item())
        loss_hist_data.append(loss_data.item())

        # Print informatie over de training en een schatting van de resterende tijd elke 100 epochs
        if epoch % 100 == 0:
            current_time = time.time()
            if epoch > 0:
                elapsed_100 = current_time - last_100_time
                remaining_epochs = EPOCHS - epoch
                estimated_time = (remaining_epochs / 100) * elapsed_100
                minutes = int(estimated_time // 60)
                seconds = int(estimated_time % 60)
                print(f"Epoch {epoch}/{EPOCHS} | Tot Loss: {total_loss.item():.2e} | PDE Loss: {loss_pde.item():.2e} | Init Loss: {loss_init.item():.2e} | Rest: {minutes}m {seconds}s")
            else:
                print(f"Epoch {epoch}/{EPOCHS} | Tot Loss: {total_loss.item():.2e} | PDE Loss: {loss_pde.item():.2e}")
                
            last_100_time = current_time

    # --- MODEL OPSLAAN ---
    if naam_model != '':
        state_path = basis_pad / naam_model
        model.to('cpu')
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), state_path)
        print("Model is opgeslagen!")
    else:
        print("Model is niet opgeslagen omdat er geen naam is opgegeven.")

    # --- PLOTS MAKEN ---
    if plot_total_loss_name != '':
        plt.rcParams.update({"font.size": 18})
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs_scaled = np.arange(len(loss_hist_tot))
        ax.semilogy(epochs_scaled, loss_hist_tot, color="black", linewidth=2)
        ax.set_xlabel("Trainingsiteratie", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.set_title("Training Loss", fontsize=16)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis='both', labelsize=12)
        fig.tight_layout()
        fig.savefig(basis_pad / (plot_total_loss_name + '.png'), dpi=200)
        ax.legend(fontsize=14)
        plt.close(fig)

    if plot_comparison_loss_name != '':
        plt.rcParams.update({"font.size": 18})
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs_scaled = np.arange(len(loss_hist_tot))
        ax.semilogy(epochs_scaled, loss_hist_tot, label="Totale Loss", color="blue", linewidth=0.8)
        ax.semilogy(epochs_scaled, loss_hist_pde, label="PDE Loss", color="green", linewidth=0.8)
        ax.semilogy(epochs_scaled, loss_hist_data, label="Data Loss", color="orange", linewidth=0.8)
        ax.semilogy(epochs_scaled, loss_hist_bdy, label="Boundary Loss", color="red", linewidth=0.8)
        ax.semilogy(epochs_scaled, loss_hist_init, label="Initial Loss", color="purple", linewidth=0.8)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlabel("Trainingsiteratie", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.set_title("Loss-componenten met Vaste Gewichten", fontsize=16)
        ax.legend(
        fontsize=14,
        labelspacing=0.2,  # Verticale ruimte tussen de verschillende labels
        handlelength=1.5,  # De lengte van de gekleurde voorbeeldlijntjes
        borderpad=0.4      # De witruimte tussen de tekst en de rand van het kader
        )
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(basis_pad / (plot_comparison_loss_name + '.png'), dpi=200)
        plt.close(fig)

    # --- RETURN WAARDEN ---
    # Flexible returns: allow combinations like ['Leersnelheid','model']
    results = []
    if 'Leersnelheid' in returns:
        def leer_curve(t, K_inf, alpha):
            return K_inf * (1 - np.exp(-alpha * t))
        t_epochs = np.arange(EPOCHS)
        if len(K_history) >= 3:
            popt, pcov = curve_fit(leer_curve, t_epochs, K_history, p0=(1.0, 0.001))
            K_inf_pred, alpha_pred = popt
        else:
            K_inf_pred, alpha_pred = 1.0, 0.0
        results.extend([K_history, K_inf_pred, alpha_pred])

    if 'losses_history' in returns:
        results.extend([loss_hist_tot, loss_hist_pde, loss_hist_init, loss_hist_bdy, loss_hist_data])

    if 'model' in returns:
        results.append(model)

    if len(results) == 0:
        return None
    if len(results) == 1:
        return results[0]
    return tuple(results)
