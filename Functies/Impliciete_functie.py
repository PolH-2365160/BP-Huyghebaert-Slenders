import numpy as np
import scipy


def impliciete_oplossing(L: float,T: float,r: float,D: float,aantal_x: int, aantal_t: int, u: np.ndarray)-> np.ndarray:
    """Berekent u^n+1 indien u is gegeven voor de Fisher-KPP vergelijking mbv een impliciete methode.

    Args:
        L (float): Lengte van het interval (het interval begint altijd bij 0)
        T (float): Laatste tijdspunt van het tijdsinterval (het interval begint altijd bij 0)
        r (float): Reactiecoëfficiënt, r > 0
        D (float): Diffusieco_efficiënt, D > 0
        aantal_x (int): Het aantal punten waarin je het afstandsinterval wilt opdelen. Moet groter zijn dan 0.
        aantal_t (int): Het aantal punten waarin je het tijdsinterval wilt opdelen. Moet groter zijn dan 0.
        u (np.ndarray): De vorige iteratieve oplossing.

    Returns:
        np.ndarray: Geeft de volgende iteratieve oplossing

    """
    # 1. Bereken del_x en del_t
    del_x = L / (aantal_x-1)
    del_t = T / aantal_t

    # 2. Maak de diagonalen van de matrix M aan
    # Boven diagonaal
    boven = -del_t*D/del_x**2 * np.ones(aantal_x-1)
    boven[0]= -2*del_t*D/del_x**2 # Randvoorwaarde links

    # Onderdiagonaal
    onder = -del_t*D/del_x**2 * np.ones(aantal_x-1)
    onder[-1]= -2*del_t*D/del_x**2 # Randvoorwaarde rechts

    # Hoofddiagonaal
    midden =  2*del_t*D/del_x**2 - (del_t*r) + 1 + del_t*r*u

    # 3. Maak de matrix M aan en los het stelsel op
    M = scipy.sparse.diags([onder, midden, boven], [-1, 0, 1], format='csr')
    u = scipy.sparse.linalg.spsolve(M,u)
    return u
