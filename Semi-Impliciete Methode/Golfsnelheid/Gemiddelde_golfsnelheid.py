import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = current_dir + "/../../"
sys.path.append(parent_dir)
from Functies.Impliciete_functie import impliciete_oplossing


###### inputwaarden geven ######
L = 100
T = 200
r = 1
D = 1
aantal_x = 10000
aantal_t = 3200
x_waarden = np.linspace(0, L, aantal_x)
t_waarden = np.linspace(0, T, aantal_t+1)
u = 1/(1+np.exp(x_waarden-10))

Theoretische_snelheid = 2 * np.sqrt(r*D)
del_x = L / (aantal_x-1)
del_t = T / aantal_t

# Definieer een functie voor interpolatie om de positie van het golffront te vinden
def interpolatie(u, index):
    while u[index] >= 0.5:
        index +=1
    punt1 = index-1
    punt2 = index
    x = (0.5 - u[punt1]) * (del_x*punt2 - del_x*punt1) / (u[punt2] - u[punt1]) + del_x*punt1
    return [x, index] 


###### Initiele berekeningen ######
aantal_plots = 400
del_x = L / (aantal_x-1)
del_t = T / aantal_t
snelheid = np.zeros(aantal_t+1)


#### Numerieke berekening #####
u_nieuw = np.empty(aantal_x)
geïnterpoleerde_posities = []
index = 0
for n in range(1,aantal_t+1):
    u_nieuw = impliciete_oplossing(L,T,r,D,aantal_x, aantal_t, u)
    huidige_t = n * del_t
    # Bereken snelheid alleen bij plot-intervallen
    if min(u_nieuw) < 0.5 and max(u_nieuw) > 0.5 and n % (aantal_t // aantal_plots) == 0:
        if index == 0:
            # interpolate1 = interpolatie(u, 0)[0]
            [interpolate2,index] = interpolatie(u_nieuw, 0)
        else:
            # interpolate1 = interpolatie(u, index)[0]
            interpolate2 = interpolatie(u_nieuw, index)[0]
            index = interpolatie(u_nieuw, index)[1]
        geïnterpoleerde_posities.append(interpolate2)

    u = u_nieuw.copy()

# Gemiddelde golfsnelheid berekenen
for i in range(1, len(geïnterpoleerde_posities)):
    delta_x = geïnterpoleerde_posities[i] - geïnterpoleerde_posities[i-1]
    plot_interval = aantal_t // aantal_plots
    delta_t = del_t * plot_interval
    snelheid[i] = delta_x / delta_t

snelheid = snelheid[snelheid != 0]


###### Resultaat van golfsnelheid plotten ######

plot_x_as = np.linspace(0, T, len(snelheid))
plt.plot(plot_x_as, snelheid, label='Gemiddelde golfsnelheid', color='blue')
plt.xlim(0, T+4)
plt.ylim(0, 4)
plt.axhline(Theoretische_snelheid, color='red', linestyle='--', label=f'Theoretische snelheid = {Theoretische_snelheid:.2f}')
plt.xlabel('Tijd (t)', fontsize=14)
plt.ylabel('Gemiddelde golfsnelheid', fontsize=14)
plt.title('Gemiddelde golfsnelheid van de golffront', fontsize=16)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=12)
plt.xticks(np.arange(0, T+1, 20), fontsize=14)
plt.yticks(np.arange(0, 4.5, 0.5), fontsize=14)
plt.savefig(current_dir + '/gemiddelde_golfsnelheid.png', dpi=300, bbox_inches='tight')
plt.show()
