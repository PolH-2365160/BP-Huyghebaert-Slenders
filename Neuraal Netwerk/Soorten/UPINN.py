import torch 
import torch.nn as nn 
import torch.optim as optim 
from pathlib import Path
import os

current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

# 1. Modeldefinitie
class SimpleUPINN(nn.Module): 
    def __init__(self): 
        super(SimpleUPINN, self).__init__() 
        self.fc1 = nn.Linear(2, 5)  # Iets breder gemaakt voor soepelere gradiënten
        self.relu = nn.ReLU()      
        self.fc2 = nn.Linear(5, 1)  

    def forward(self, x): 
        x = self.fc1(x) 
        x = self.relu(x) 
        x = self.fc2(x) 
        return x 

def train_upinn_model(epochs=2500, lr=0.01, num_collocation=50):
    """
    Traint een UPINN (Unsupervised Physics-Informed Neural Network) model en returnt het.
    
    Args:
        epochs: Aantal trainingsiteraties (default: 2500)
        lr: Leersnelheid (default: 0.01)
        num_collocation: Aantal collocation punten per epoch (default: 50)
    
    Returns:
        Het getrainde UPINN model
    """
    # Model aanmaken
    model = SimpleUPINN()
    
    # Training Setup
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    
    # Randvoorwaarde (Boundary Data) om de constante '+ 2' te bepalen
    bc_input = torch.tensor([[0.0, 0.0]]) 
    bc_target = torch.tensor([[2.0]]) 
    
    # Loss functie
    mse_loss = nn.MSELoss()
    
    # Trainen
    for epoch in range(epochs): 
        optimizer.zero_grad() 
    
        # --- A. Boundary Loss ---
        # Controleer of het model f(0,0) = 2 respecteert
        bc_output = model(bc_input)
        loss_boundary = mse_loss(bc_output, bc_target)
    
        # --- B. Physics Loss ---
        # We genereren willekeurige punten in het domein om de "fysica" te testen
        collocation_points = (torch.rand(num_collocation, 2) * 10.0).requires_grad_(True)
        
        outputs = model(collocation_points)
        
        # Bereken de partiële afgeleiden van de output t.o.v. de inputs (a en b)
        gradients = torch.autograd.grad(
            outputs=outputs,
            inputs=collocation_points,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True 
        )[0]
        
        df_da = gradients[:, 0:1] # De afgeleide naar a
        df_db = gradients[:, 1:2] # De afgeleide naar b
        
        # De "fysica" eist dat beide afgeleiden exact 1.0 moeten zijn
        loss_physics = mse_loss(df_da, torch.ones_like(df_da)) + \
                       mse_loss(df_db, torch.ones_like(df_db))
    
        # --- C. Totale Loss ---
        # Combineer de loss van de randvoorwaarde met de loss van de fysica
        loss = loss_boundary + loss_physics 
        
        loss.backward() 
        optimizer.step()
    
    return model

if __name__ == "__main__":
    # Model trainen
    model = train_upinn_model()
    
    # Testen
    test_data = torch.tensor([[4.0, 5.0]]) 
    prediction = model(test_data) 
    print(f'UPINN voorspelling voor [4.0, 5.0]: {prediction.item():.10f}')