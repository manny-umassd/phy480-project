import torch  # Import PyTorch library for building and training the neural network.
from torch.utils.data import Dataset, DataLoader  # Utilities for dataset handling and batching.
import pandas as pd  # Pandas for data manipulation and analysis.
import numpy as np  
import os 


# Dataset for Solar System Simulation Data 
class SolarSystemDataset(Dataset):
    
    #Initialize the dataset by loading data from a CSV file.
    #csv_file: Path to the CSV file containing simulation data.
    def __init__(self, csv_file):
       
        # Load the data from the CSV file into a Pandas Dataframe.
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
       
        # Return the total number of samples in the dataset.
        
        # Using length-1 because I need a "next state" for each sample.
        return len(self.data) - 1

    def __getitem__(self, idx):
        
        # Retrieve a single sample from the dataset.
        # idx: Index of the sample to retrieve.
        # return: Input (current state) and target (next state).
        
        # Current state (all columns except time in column 0).
        current_state = self.data.iloc[idx, 1:].values.astype(np.float32)
        
        # Next state (the next row in the dataset).
        next_state = self.data.iloc[idx + 1, 1:].values.astype(np.float32)

        return torch.tensor(current_state), torch.tensor(next_state)


# Define the Neural Network 
class PINN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=3, hidden_units=128):
        
        # Initialize the PINN model.
        # input_size: Number of input features (positions and velocities).
        # output_size: Number of output features (positions and velocities).
        # hidden_layers: Number of hidden layers in the neural network.
        # hidden_units: Number of neurons per hidden layer.
        
        super(PINN, self).__init__()

        # Layers of the model
        layers = []

        # Input layer
        layers.append(torch.nn.Linear(input_size, hidden_units))
        layers.append(torch.nn.ReLU())

        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(torch.nn.Linear(hidden_units, hidden_units))
            layers.append(torch.nn.ReLU())

        # Output layer
        layers.append(torch.nn.Linear(hidden_units, output_size))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        
        #Forward pass through the network.
        # x: Input tensor (current state).
        # return: Output tensor (predicted next state).
        
        return self.model(x)


# Define the Loss Function returning both data_loss and physics_loss
def compute_loss(predictions, targets, positions, velocities, masses, G=4*(np.pi**2), Δt=0.001):
    
    # Compute both data-driven (MSE) and physics-based losses.
    # MSE, or Mean Squared Error is a loss function in Machine Learning, it measures the average of the squares of the errors. The difference between predicted values and target values.
    # Physics loss is essentially the P(hysics) of the PINN model. The Loss component that relies on physical equations/laws.
    
    # predictions: Predicted next state (positions+velocities).
    # targets: True next state (positions+velocities) based on simulation.
    # positions: Current positions in AU.
    # velocities: Current velocities in AU/yr.
    # masses: Masses of the bodies in Solar Masses.
    # G: Gravitational constant
    # Δt: Time step in years. (Δ is 100% pasted in, I can't keyboard that in except by ctrl+v but it's easier to look at)
    #return: (data_loss, physics_loss)
   
    # Data-driven MSE loss
    data_loss = torch.nn.functional.mse_loss(predictions, targets)

    batch_size, total_features = predictions.shape
    num_bodies = total_features // 6  # each body: 3 pos + 3 vel
    pos_dim = num_bodies * 3 #pos dimensions
    vel_dim = num_bodies * 3 #vel dimensions

    # Extract predicted positions and velocities
    predicted_positions = predictions[:, :pos_dim]
    predicted_velocities = predictions[:, pos_dim:pos_dim+vel_dim]

    # Compute predicted accelerations
    predicted_accelerations = (predicted_velocities - velocities) / Δt

    physics_loss = 0.0
    for i in range(num_bodies):
        force_residual = torch.zeros_like(predicted_positions[:, i*3:(i+1)*3])
        for j in range(num_bodies):
            if i != j:
                r_ij = positions[:, i*3:(i+1)*3] - positions[:, j*3:(j+1)*3]
                # Clamping the distance to a minimum of 1 to avoid huge forces exploding the training loss. (Had happened.)
                dist = torch.clamp(torch.norm(r_ij, dim=1, keepdim=True), 1)
                force = -G * masses[j] * r_ij / (dist**3)
             
                force_residual += force

        # Compute physics loss for body i
        physics_loss += torch.mean((predicted_accelerations[:, i*3:(i+1)*3] - force_residual)**2)

    # Optional: normalize physics_loss by num_bodies if desired
    physics_loss = physics_loss / num_bodies

    return data_loss, physics_loss


# MAIN SCRIPT: Training Loop with Gradual Physics Introduction

if __name__ == "__main__":
    # Set working directory
    os.chdir("C:\\Users\\Manny Admin\\Desktop\\New Data\\Simulation Pull")

    # CSV file path
    csv_file = "simulation_data.csv"

    # Initialize dataset
    dataset = SolarSystemDataset(csv_file)

    # Split dataset into training (80%) and validation (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Determine input/output size
    sample_input, sample_target = dataset[0]
    input_size = len(sample_input)
    output_size = len(sample_target)

    # Initialize model
    model = PINN(input_size, output_size)
    print(model)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) #Adjust Learning Rate Here

    # Training parameters
    epochs = 200          #Adjust epochs here, 200 is NOT enough, try 3500  (Well it might be with longer dataset)
    masses = torch.tensor([1.0, 3.004e-6, 9.551e-4], dtype=torch.float32)

    # Physics introduction parameters
    start_physics_epoch = 10
    max_physics_weight = 1e-4 #Adjust weight of physics laws on training here. 0 physics creates the BlackBox model. 
    physics_weight = 0.0 # Initial physics weight, allows the model to get a grasp of the data and adjust smoother.

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0

        # After start_physics_epoch, gradually increase physics weight
        if epoch > start_physics_epoch:
            physics_weight = (epoch - start_physics_epoch) * (max_physics_weight / 10.0)
            physics_weight = min(physics_weight, max_physics_weight)
        else:
            physics_weight = 0.0

        for batch_idx, batch in enumerate(train_loader):
            current_state, next_state = batch

            num_bodies = input_size // 6
            pos_dim = num_bodies * 3
            vel_dim = num_bodies * 3

            positions = current_state[:, :pos_dim]
            velocities = current_state[:, pos_dim:pos_dim+vel_dim] # will do a deeper dive in notebook on the training loop
                                                                   # and add comments along the way, for now guestimate based on labels
            predictions = model(current_state)

            data_loss, physics_loss = compute_loss(predictions, next_state, positions, velocities, masses)

            total_loss = data_loss + physics_weight * physics_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += total_loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                current_state, next_state = batch
                positions = current_state[:, :pos_dim]
                velocities = current_state[:, pos_dim:pos_dim+vel_dim]
                predictions = model(current_state)
                data_loss_val, physics_loss_val = compute_loss(predictions, next_state, positions, velocities, masses)
                val_total_loss = data_loss_val + physics_weight * physics_loss_val
                val_loss_sum += val_total_loss.item()

        avg_val_loss = val_loss_sum / len(val_loader)
        
        #print function to visualize progress. (Fun tidbit. Losses once began in the octillions because I added a safeguard against division by 0. All I had to do was let the model know those columns were supposed to be 0 entirely)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Physics Weight: {physics_weight:.2e}")

    # Save the trained model to my usual directory
    model_save_path = os.path.join("C:\\Users\\Manny Admin\\Desktop\\New Data\\Simulation Pull", "PINN_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
