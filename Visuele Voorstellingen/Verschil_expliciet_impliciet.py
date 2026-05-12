import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# --- 1. Systeempaden en Imports ---
current_dir = Path(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else Path.cwd()
root_dir = current_dir.parents[0] # Gaat één map omhoog naar de root

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Importeer de functies uit de nieuwe mappenstructuur
from Functies.Expliciete_functie import expliciete_oplossing
from Functies.Impliciete_functie import impliciete_oplossing

# Maak de outputmap aan voor de figuren
Output_dir = current_dir / 'Figuren'
Output_dir.mkdir(parents=True, exist_ok=True)

# --- 2. Instellen van de parameters ---
L = 20.0          # Lengte van het x-interval
T = 10.0          # Totale simulatietijd
r = 1.0           # Groeifactor
D = 1             # Diffusiecoëfficiënt
aantal_x = 200     # Aantal ruimtestappen
aantal_t = 3200   # Aantal tijdstappen (voldoet aan stabiliteit voor expliciet)

# --- 3. Initialisatie ---
x_waarden = np.linspace(0, L, aantal_x)
# Beginvoorwaarde: Logistische curve/golf
u0 = 1 / (1 + np.exp(x_waarden - 10)) 

# Matrices initialiseren om de volledige tijdsevolutie op te slaan
u_mat_exp = np.zeros((aantal_t + 1, aantal_x))
u_mat_imp = np.zeros((aantal_t + 1, aantal_x))

# Begincondities invullen
u_mat_exp[0] = u0
u_mat_imp[0] = u0

# Tijdelijke vectoren voor de lus
u_curr_exp = u0.copy()
u_curr_imp = u0.copy()

# --- 4. Tijdslus ---
# Beide methoden worden in dezelfde lus berekend voor maximale efficiëntie
for n in range(1, aantal_t + 1):
    u_curr_exp = expliciete_oplossing(L, T, r, D, aantal_x, aantal_t, u_curr_exp)
    u_curr_imp = impliciete_oplossing(L, T, r, D, aantal_x, aantal_t, u_curr_imp)
    
    u_mat_exp[n] = u_curr_exp
    u_mat_imp[n] = u_curr_imp

# Bereken het absolute verschil tussen beide methoden
verschil = np.abs(u_mat_exp - u_mat_imp)

# --- 5. Visualisatie: Heatmaps ---
plt.figure(figsize=(14, 5))

# Plot 1: Expliciete Oplossing
plt.subplot(1, 3, 1)
plt.imshow(u_mat_exp,   extent=(0.0, L, 0.0, T), aspect='auto', origin='lower', cmap='viridis')
plt.title("Expliciete Oplossing", fontsize=14)
plt.ylabel("Tijd (t)", fontsize=14)
plt.xlabel("Positie (x)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
cbar = plt.colorbar()
cbar.set_label('Oplossing u(x,t)', fontsize=14)
cbar.ax.tick_params(labelsize=12)


# Plot 2: Kwadratisch Verschil
plt.subplot(1, 3, 2)
plt.imshow(verschil, extent=(0.0, L, 0.0, T), aspect='auto', origin='lower', cmap='viridis')
plt.title("Absoluut Verschil", fontsize=14)
plt.ylabel("Tijd (t)", fontsize=14)
plt.xlabel("Positie (x)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
cbar2 = plt.colorbar()
cbar2.set_label('Absoluut Verschil', fontsize=14)
cbar2.ax.tick_params(labelsize=12)


# Plot 3: Impliciete Oplossing
plt.subplot(1, 3, 3)
plt.imshow(u_mat_imp, extent=(0.0, L, 0.0, T), aspect='auto', origin='lower', cmap='viridis')
plt.title("Semi-impliciete Oplossing", fontsize=14)
plt.ylabel("Tijd (t)", fontsize=14)
plt.xlabel("Positie (x)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
cbar3 = plt.colorbar()
cbar3.set_label('Oplossing u(x,t)', fontsize=14)
cbar3.ax.tick_params(labelsize=12)


plt.tight_layout()

# Opslaan van de figuur in de nieuwe mappenstructuur
save_path = Output_dir / 'Vergelijking_Expliciet_Impliciet_Heatmap.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Figuur succesvol opgeslagen in: {save_path}")

plt.show()