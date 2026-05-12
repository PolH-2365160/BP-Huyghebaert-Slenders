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

# Importeer de impliciete functie
from Functies.Impliciete_functie import impliciete_oplossing

Output_dir = current_dir / 'Figuren'
Output_dir.mkdir(parents=True, exist_ok=True)

# --- 2. Instellen van de parameters ---
L = 30.0          # Lengte van het x-interval
T = 10.0          # Totale simulatietijd 
r = 1.0           # Groeifactor
D = 1             # Diffusiecoëfficiënt
aantal_x = 200    # Aantal ruimtestappen 
aantal_t = 3200   # Aantal tijdstappen

# Berekening voor de lijn-plots: na hoeveel stappen er 2 seconden voorbij zijn
tijd_tussen_plots = 2.0
stappen_per_plot = int((tijd_tussen_plots / T) * aantal_t)

# --- 3. Initialisatie ---
x_waarden = np.linspace(0, L, aantal_x)
u_imp = 1 / (1 + np.exp(x_waarden - 10)) # Beginvoorwaarde

# Maak een 2D-matrix om alle tijdstappen in op te slaan
u_matrix = np.zeros((aantal_t + 1, aantal_x))
u_matrix[0] = u_imp 

# --- 4. Berekening (Tijdslus) ---
# We vullen de matrix hier volledig in zodat we deze voor beide plots kunnen gebruiken
for n in range(1, aantal_t + 1):
    u_imp = impliciete_oplossing(L, T, r, D, aantal_x, aantal_t, u_imp)
    u_matrix[n] = u_imp

# --- 5. Visualisatie Deel 1: Heatmap Plotten ---
plt.figure(figsize=(10, 6), num = 1)

# origin='lower' zorgt dat t=0 onderaan begint
# extent koppelt de matrix-indices aan de werkelijke L en T waarden
plt.imshow(u_matrix, origin='lower', extent=(0.0, L, 0.0, T), aspect='auto', cmap='viridis')

# Kleurenbalk en labels
cbar = plt.colorbar()
cbar.set_label('Oplossing u(x,t)', fontsize=14)
cbar.ax.tick_params(labelsize=12)

plt.title('Semi-impliciete Oplossing (Heatmap)', fontsize=14)
plt.xlabel('Positie (x)', fontsize=14)
plt.ylabel('Tijd (t)', fontsize=14)

# Streepjes op de assen
plt.xticks(np.arange(0, L+1, 2), fontsize=12)
plt.yticks(np.arange(0, T+1, 1), fontsize=12) 

plt.tight_layout()
plt.savefig(Output_dir / 'Heatmap_impliciete_oplossing.png', dpi=300, bbox_inches='tight')

# --- 6. Visualisatie Deel 2: Lijn-grafiek (Lopende Golf) ---
plt.figure(figsize=(10, 6), num = 2)

# Plot de initiële toestand (t=0) uit de eerste rij van de matrix
plt.plot(x_waarden, u_matrix[0], label='t = 0 s', linestyle=':', color='gray', linewidth=2)

# Loop door de matrix met de stapgrootte die we eerder hebben berekend
for n in range(1, aantal_t + 1):
    if n % stappen_per_plot == 0:
        huidige_tijd = (n / aantal_t) * T
        plt.plot(x_waarden, u_matrix[n], label=f't = {int(huidige_tijd)} s', linewidth=2)

# Opmaak van de lijn-grafiek
plt.title('Lopende Golf-oplossing berekend d.m.v. Semi-impliciete EDM', fontsize=14)
plt.xlabel('Positie (x)', fontsize=14)
plt.ylabel('Oplossing u(x,t)', fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7)
plt.xticks(np.arange(0, L+1, 5), fontsize=14)
plt.yticks(np.arange(0, 1.2, 0.2), fontsize=14)

plt.legend(fontsize=12, loc='upper right')
plt.tight_layout()
plt.savefig(Output_dir / 'Lopende_golfoplossing_impliciete_methode.png', dpi=300, bbox_inches='tight')
plt.show()