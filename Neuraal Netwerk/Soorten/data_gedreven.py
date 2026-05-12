import torch 
import torch.nn as nn 
import torch.optim as optim 
from pathlib import Path
import os

# 1. Paden instellen
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

# 2. Modeldefinitie
class SimpleNN(nn.Module): 
  def __init__(self): 
    super(SimpleNN, self).__init__() 
    self.fc1 = nn.Linear(2, 5)   
    self.relu = nn.ReLU()      # Activatiefunctie
    self.fc2 = nn.Linear(5, 1)  

  def forward(self, x): 
    x = self.fc1(x) 
    x = self.relu(x) 
    x = self.fc2(x) 
    return x 

def train_data_gedreven_model(epochs=2500, lr=0.01):
  """
  Traint een data-gedreven model en returnt het.
  
  Args:
    epochs: Aantal trainingsiteraties (default: 2500)
    lr: Leersnelheid (default: 0.01)
  
  Returns:
    Het getrainde model
  """
  # Model aanmaken
  model = SimpleNN() 

  # Training Setup
  criterion = nn.MSELoss() 
  optimizer = optim.SGD(model.parameters(), lr=lr) 

  # Trainingsdata
  inputs = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [5.0, 6.0]]) 
  targets = torch.tensor([[5.0], [7.0], [9.0], [13.0]]) 

  # Trainen
  for epoch in range(epochs): 
    optimizer.zero_grad()               # Vorige gradienten resetten 
    outputs = model(inputs)             # Voorwaartse pass
    loss = criterion(outputs, targets)  # Bereken loss 
    loss.backward()                     # Backpropagation
    optimizer.step()                    # Update de gewichten
  
  return model

if __name__ == "__main__":
  # Model trainen
  model = train_data_gedreven_model()
  
  # Testen
  test_data = torch.tensor([[4.0, 5.0]]) 
  prediction = model(test_data) 
  print(f'Data-gedreven voorspelling voor [4.0, 5.0]: {prediction.item():.10f}') 
