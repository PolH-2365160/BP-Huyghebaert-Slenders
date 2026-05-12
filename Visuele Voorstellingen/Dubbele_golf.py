import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# --- 1. Systeempaden en Imports ---
current_dir = Path(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else Path.cwd()
root_dir = current_dir.parents[0]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Importeer de impliciete functie uit de nieuwe locatie
from Functies.Impliciete_functie import impliciete_oplossing

# Maak de outputmap aan voor de figuren
Output_dir = current_dir / 'Figuren'
Output_dir.mkdir(parents=True, exist_ok=True)

# --- 2. Instellen van de parameters ---
L = 20.0          # Lengte van het x-interval
T = 10.0          # Totale simulatietijd 
r = 1.0           # Groeifactor
D = 1             # Diffusiecoëfficiënt
aantal_x = 200    # Aantal ruimtestappen 
aantal_t = 3200   # Aantal tijdstappen

# Voor de lijnplot: bereken na hoeveel stappen er 2 seconden voorbij zijn
tijd_tussen_plots = 2.0
stappen_per_plot = int((tijd_tussen_plots / T) * aantal_t)

# --- 3. Initialisatie ---
x_waarden = np.linspace(0, L, aantal_x)
# Beginvoorwaarde: Dubbele golf
u_curr = 1 / (1 + np.exp(x_waarden - 2)) + 1 / (1 + np.exp(-(x_waarden - 18))) 

# Maak een 2D-matrix om alle data in op te slaan voor de heatmap
u_matrix = np.zeros((aantal_t + 1, aantal_x))
u_matrix[0] = u_curr 

# --- 4. Tijdslus ---
for n in range(1, aantal_t + 1):
    u_curr = impliciete_oplossing(L, T, r, D, aantal_x, aantal_t, u_curr)
    u_matrix[n] = u_curr

# --- 5. Visualisatie Deel 1: Heatmap Plotten (Exacte opmaak behouden) ---
plt.figure(figsize=(10, 6), num=1)

plt.imshow(u_matrix, origin='lower', extent=(0.0, L, 0.0, T), aspect='auto', cmap='viridis')

cbar = plt.colorbar()
cbar.set_label('Oplossing u(x,t)', fontsize=14)
cbar.ax.tick_params(labelsize=14)

plt.title('Semi-impliciete Oplossing met $u_0 = \\frac{1}{1+e^{x-2}} + \\frac{1}{1+e^{-x+18}}$', fontsize=14)
plt.xlabel('Positie (x)', fontsize=14)
plt.ylabel('Tijd (t)', fontsize=14)
plt.xticks(np.arange(0, L+1, 2.5), fontsize=14)
plt.yticks(np.arange(0, T+1, 2), fontsize=14) 

plt.tight_layout()
plt.savefig(Output_dir / 'heatmap_dubbele_golf.png', dpi=300, bbox_inches='tight')

# --- 6. Visualisatie Deel 2: Lijnplot (Exacte opmaak uit vorige stap) ---
plt.figure(figsize=(10, 6), num=2)

# Plot de initiële toestand (t=0)
plt.plot(x_waarden, u_matrix[0], label='t = 0 s', linestyle=':', color='gray', linewidth=2)

# Gebruik de data uit de matrix voor de lijnen per tijdstap
for n in range(1, aantal_t + 1):
    if n % stappen_per_plot == 0:
        huidige_tijd = (n / aantal_t) * T
        plt.plot(x_waarden, u_matrix[n], label=f't = {int(huidige_tijd)} s', linewidth=2)

plt.title('Evolutie Lopende Golf-oplossing voor Fisher-KPP-vergelijking met $u_0(x) = \\frac{1}{1 + e^{-x}} + \\frac{1}{1 + e^{-(x-18)}}$', fontsize=14)
plt.xlabel('Positie (x)', fontsize=14)
plt.ylabel('Oplossing u(x,t)', fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7)
plt.xticks(np.arange(0, L+1, 5), fontsize=14)
plt.yticks(np.arange(0, 1.2, 0.2), fontsize=14)
plt.legend(fontsize=12, loc='lower right') 

plt.tight_layout()
plt.savefig(Output_dir / 'lopende_golf_dubbele_golf.png', dpi=300, bbox_inches='tight')
plt.show()