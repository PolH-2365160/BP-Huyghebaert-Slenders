import torch
import torch.nn as nn
from pathlib import Path
import os 
from UPINN import train_upinn_model
from data_gedreven import train_data_gedreven_model
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))



# 1. Definieer de structuren (nodig om opgeslagen modellen in te laden)
class SimpleNN(nn.Module): 
    def __init__(self): 
        super(SimpleNN, self).__init__() 
        self.fc1 = nn.Linear(2, 5)   
        self.relu = nn.ReLU()      
        self.fc2 = nn.Linear(5, 1)  

    def forward(self, x): 
        x = self.fc1(x) 
        x = self.relu(x) 
        x = self.fc2(x) 
        return x 


# 2. Initialiseer de modellen
data_gedreven_model = train_data_gedreven_model()
upinn_model = train_upinn_model()

 
# Zorg ervoor dat beide modellen in evaluatiemodus staan
data_gedreven_model.eval()
upinn_model.eval()

# 3. Maak testdata aan: we testen binnen en ver buiten de trainingsdata
# Het data-gedreven model is getraind met inputs tussen 1.0 en 6.0
test_cases = {
    "Binnen het domein": [4.0, 5.0],
    "Net erbuiten": [8.0, 10.0],
    "Ver erbuiten": [50.0, 50.0],
    "Extreem ver": [1000.0, 2000.0]
}

# 4. Voer de test uit en print de resultaten
print("Vergelijking: Data-Gedreven NN vs UPINN")
print("-" * 60)

for beschrijving, waarden in test_cases.items():
    a, b = waarden
    
    # Bereken de ware wiskundige uitkomst: f(a,b) = a + b + 2
    verwacht = a + b + 2.0
    
    # Maak een tensor van de input
    input_tensor = torch.tensor([[a, b]])
    
    # Haal de voorspellingen op (met torch.no_grad() omdat we niet trainen)
    with torch.no_grad():
        pred_data = data_gedreven_model(input_tensor).item()
        pred_upinn = upinn_model(input_tensor).item()
    
    # Print de resultaten
    print(f"Scenario: {beschrijving}")
    print(f"Input [a, b]  : [{a}, {b}]")
    print(f"Ware uitkomst : {verwacht:.2f}")
    print(f"Voorspelling Data-NN : {pred_data:.2f} (Fout: {abs(verwacht - pred_data):.2f})")
    print(f"Voorspelling UPINN    : {pred_upinn:.2f} (Fout: {abs(verwacht - pred_upinn):.2f})")
    print("-" * 60)