# Bachelorproef: Modellering van wondsluiting via de Fisher-KPP-vergelijking

Dit repository bevat de broncode en resultaten van de bachelorproef **"Modellering van wondsluiting via de Fisher-KPP-vergelijking: een Physics-Informed Neural Network als alternatief voor de eindige-differentiemethode"**.

## 📖 Projectbeschrijving
In dit onderzoek wordt het proces van wondsluiting wiskundig gemodelleerd. De kern van het project is de vergelijking tussen twee numerieke benaderingen om de Fisher-KPP-vergelijking op te lossen:
1.  **Eindige-differentiemethode (EDM):** Een klassieke numerieke benadering (semi-impliciet) die dient als betrouwbare referentie.
2.  **Physics-Informed Neural Networks (PINNs):** Een moderne machine learning aanpak waarbij de fysische wetten (PDE's) direct in de verliesfunctie van het neuraal netwerk worden geïntegreerd.

Het doel is om te evalueren in hoeverre PINNs een efficiënt en nauwkeurig alternatief vormen voor traditionele methoden bij het simuleren van weefselherstel.

## 👥 Auteurs
* **Pol Huyghebaert** – Universiteit Hasselt
* **Goele Slenders** – Universiteit Hasselt

**Begeleiders:**
* De heer Wilbert den Hertog
* Prof. dr. Fred Vermolen

## 🛠️ Technologieën
De code is geschreven in **Python** en maakt gebruik van de volgende bibliotheken:
* `NumPy` & `SciPy` (voor de EDM simulaties)
* `PyTorch` of `TensorFlow` (voor de PINN architectuur)
* `Matplotlib` (voor visualisatie van de lopende golven)

## 🎓 Context
Deze bachelorproef is uitgevoerd aan de **Faculteit Wetenschappen** van de **Universiteit Hasselt** binnen de opleiding **Bachelor in de Wiskunde**, academiejaar 2025-2026.
