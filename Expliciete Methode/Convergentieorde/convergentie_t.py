import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = current_dir + "/../../"
sys.path.append(parent_dir)
from Functies.Expliciete_functie import expliciete_oplossing

# 1. Parameters
x_waarden = np.linspace(0,10,100)
L = 10
T = 20
r = 1
D = 1 
aantal_t = [50000,25000,12500] # Zorg dat deze telkens de helft is van de vorige
aantal_x = 150 # Kies deze zodat het voldoet aan de stabiliteitsvoorwaarde

# 2. Bereken de oplossingen voor verschillende tijdstappen
x_waarden = np.linspace(0, L, aantal_x)
u_waarden = {}
for index, aant_t in enumerate(aantal_t):
    delt = T/aant_t
    n = 0
    t_waarden = np.linspace(0, T, aant_t+1)
    u = 0.5 * (1 - np.tanh(x_waarden / 2)) # Beginvoorwaarde
    while n*delt <= 2:
        u = expliciete_oplossing(L,T,r,D,aantal_x,aant_t, u)
        n += 1
    u_waarden[aant_t] = u

# 3. Bereken de convergentieorde m.b.v. Richardson extrapolatie
E2 = np.linalg.norm(u_waarden[aantal_t[1]] - u_waarden[aantal_t[0]], 2)
E4 = np.linalg.norm(u_waarden[aantal_t[2]] - u_waarden[aantal_t[1]], 2)
p = np.log2(E4/E2)
print(f"Convergentieorde: {p:.3f}")