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
aantal_x = 20     # Aantal ruimtestappen (del_x = 1.0)
aantal_t = 3200   # Aantal tijdstappen

# --- 3. Initialisatie van het rooster en de beginvoorwaarden ---
x_waarden = np.linspace(0, L, aantal_x)

# Definiëren van de twee verschillende beginvoorwaarden
u0 = 1 / (1 + np.exp(x_waarden - 10)) 
u0_verschillend = 0.5 * (1 - np.tanh(x_waarden / 2))

# Kopieer de beginvoorwaarden voor de simulaties
u_1 = u0.copy()
u_2 = u0_verschillend.copy()

# Matrices om de evolutie in op te slaan
u_mat_1 = np.zeros((aantal_t + 1, aantal_x))
u_mat_2 = np.zeros((aantal_t + 1, aantal_x))

u_mat_1[0] = u_1
u_mat_2[0] = u_2

# --- 4. Tijdslus ---
for n in range(aantal_t):
    u_1 = impliciete_oplossing(L, T, r, D, aantal_x, aantal_t, u_1)
    u_2 = impliciete_oplossing(L, T, r, D, aantal_x, aantal_t, u_2)
    u_mat_1[n+1] = u_1
    u_mat_2[n+1] = u_2

# --- 5. Resultaten Plotten (Exacte plotcode behouden) ---
plt.figure(figsize=(15, 5))

# Plot 1: Eerste beginvoorwaarde
plt.subplot(1, 2, 1)
plt.imshow(u_mat_1, extent=(0.0, L, 0.0, T), aspect='auto', origin='lower', cmap='viridis')
plt.title(r"Semi-impliciete oplossing met $u_0 = \frac{1}{1+e^{x-10}}$", fontsize=14)
plt.ylabel("Tijd (t)", fontsize=14)
plt.xlabel("Positie (x)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
cbar = plt.colorbar()
cbar.set_label('Oplossing u(x,t)', fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Plot 2: Tweede beginvoorwaarde
plt.subplot(1, 2, 2)
plt.imshow(u_mat_2, extent=(0.0, L, 0.0, T), aspect='auto', origin='lower', cmap='viridis')
plt.title(r"Semi-impliciete oplossing met $u_0 = 0.5(1 - \tanh(\frac{x}{2}))$", fontsize=14)
plt.ylabel("Tijd (t)", fontsize=14)
plt.xlabel("Positie (x)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
cbar3 = plt.colorbar()
cbar3.set_label('Oplossing u(x,t)', fontsize=14)
cbar3.ax.tick_params(labelsize=12)

plt.tight_layout()

# Opslaan van het resultaat
plt.savefig(Output_dir / 'Vergelijking_Beginvoorwaarden_Heatmap.png', dpi=300, bbox_inches='tight')
plt.show()