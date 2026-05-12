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
basis_intervallen = 30
intervallen = [basis_intervallen*4, basis_intervallen*2, basis_intervallen]
aantal_x = [n + 1 for n in intervallen] 
aantal_t = 10000 # Kies deze zodat aan de stabiliteitsvoorwaarde wordt voldaan
u_waarden = {}
t_waarden = np.linspace(0, T, aantal_t+1)
delt = T/aantal_t

# 2. Bereken de oplossingen voor verschillende ruimtelijke stappen
for index, aant_x in enumerate(aantal_x):
    delx = L/(aant_x - 1)
    n = 0
    x_waarden = np.linspace(0, L, aant_x)
    u = 0.5 * (1 - np.tanh(x_waarden / 2))
    while n*delt <= 2:
        u = expliciete_oplossing(L,T,r,D,aant_x,aantal_t, u)
        n += 1
    u_waarden[aant_x] = u

# 3. Bereken de convergentieorde m.b.v. Richardson extrapolatie
E2 = np.linalg.norm(u_waarden[aantal_x[1]][::2] - u_waarden[aantal_x[0]][::4], 2)
E4 = np.linalg.norm(u_waarden[aantal_x[2]] - u_waarden[aantal_x[1]][::2], 2)
p = np.log2(E4/E2)
print(f"Convergentieorde: {p:.3f}")