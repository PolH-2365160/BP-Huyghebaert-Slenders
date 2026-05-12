import numpy as np


def expliciete_oplossing(L: float,T: float,r: float,D: float,aantal_x: int, aantal_t: int, u: np.ndarray)-> np.ndarray:
    """ Functie die de expliciete oplossing berekent

    Args:
        L (float): Lengte van x-interval
        T (float): Tijd-interval
        r (float): Groeifactor
        D (float): Diffusiecoefficient
        aantal_x (int): Aantal x-stappen
        aantal_t (int): Aantal t-stappen
        u (np.ndarray): oplossing van de vorige tijdstap

    Raises:
        ValueError: Als de tijdstap te groot is voor stabiliteit van de expliciete methode.

    Returns:
        np.ndarray: Oplossing van de volgende tijdstap
    """
    # 1. Bereken del_t en del_x
    del_t = T / aantal_t
    del_x = L / (aantal_x - 1)

    # 2. Controleer stabiliteit van de expliciete methode
    if del_t > 2/(4*D/del_x**2 + r):
        print(f"Het werkt niet met aantal_t = {aantal_t}")
        raise ValueError("Tijdstap is te groot voor stabiliteit van de expliciete methode.")
    
    # 3. Bereken de volgende tijdstap
    u_nieuw = np.empty(aantal_x)
    u_nieuw[0] = del_t*(2*D*(u[1]-u[0])/del_x**2 + r*u[0]*(1-u[0])) + u[0] # Randvoorwaarde links
    for i in range(1, aantal_x-1):
        u_nieuw[i]     = del_t*(D*(u[i+1]-2*u[i]+u[i-1])/del_x**2 + r*u[i]*(1-u[i])) + u[i]
    u_nieuw[aantal_x-1] = del_t*(2*D*(u[aantal_x-2]-u[aantal_x-1])/del_x**2 + r*u[aantal_x-1]*(1-u[aantal_x-1])) + u[aantal_x-1] # Randvoorwaarde rechts
    return u_nieuw