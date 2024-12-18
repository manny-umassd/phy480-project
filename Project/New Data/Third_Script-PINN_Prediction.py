import numpy as np
from vpython import vector, sphere, canvas, rate, textures, color
import torch
import os

# No gravitational constants or masses needed directly now, since we use the model (BlackBox model reference)
class PINN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=3, hidden_units=128):
        
        #Initialize the PINN model
        #input_size: Number of input features (positions and velocities).
        #output_size: Number of output features (positions and velocities).
        #hidden_layers: Number of hidden layers in the neural network.
        #hidden_units: Number of neurons per hidden layer.
        
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
        #Input tensor (current state).
        #Output tensor (predicted next state).
        
        return self.model(x)

bodies = [   #Can add others if others are available. Match with sim info.
    {
        'name': 'Sun',
        'mass': 1.0,
        'pos': vector(0, 0, 0),
        'vel': vector(0, 0, 0),
        'radius': 0.2,
        'color': color.yellow,
        'texture': None,
    },
    {
        'name': 'Earth',
        'mass': 3.0e-6,
        'pos': vector(1.0, 0, 0),  
        'vel': vector(0, 0, -6.179),  
        'radius': 0.1,
        'color': color.white,
        'texture': textures.earth,
    },
    {
        'name': 'Jupiter',
        'mass': 9.5e-4,
        'pos': vector(5.2, 0, 0),  
        'vel': vector(0, 0, -2.624),  
        'radius': 0.15,
        'color': color.orange,
        'texture': None,
    },
]

def load_model(model_path):
    
    # Load the pre-trained PINN model from a .pth file.
    
    # Define the model architecture (ensure this matches the training script)
    input_size = 18  # This must match the dataset's input size
    output_size = 18  # This must match the dataset's output size
    model = PINN(input_size=input_size, output_size=output_size, hidden_layers=3, hidden_units=128)

    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()
    return model

def get_state_vector(bodies):
   
    # Construct the state vector (positions and velocities) from the list of bodies.
    # Order: [x,y,z,vx,vy,vz] for each body in the order they are listed in 'bodies'.
    
    state = []
    for b in bodies:
        state.append(b['pos'].x)
        state.append(b['pos'].y)
        state.append(b['pos'].z)
        state.append(b['vel'].x)
        state.append(b['vel'].y)
        state.append(b['vel'].z)
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # shape [1, features]

def update_bodies_from_state(bodies, state):
   
    # Update the positions and velocities of the bodies from the given state vector.
    # state is a torch tensor of shape [1, total_features], same order as get_state_vector.
    
    state = state.squeeze(0).detach().numpy()  # convert to numpy, shape [features]
    num_bodies = len(bodies)
    for i, b in enumerate(bodies):
        idx = i * 6
        b['pos'].x = state[idx]
        b['pos'].y = state[idx+1]
        b['pos'].z = state[idx+2]
        b['vel'].x = state[idx+3]
        b['vel'].y = state[idx+4]
        b['vel'].z = state[idx+5]

def simulate():
    # Set the working directory to where model and data are located
    os.chdir("C:\\Users\\Manny Admin\\Desktop\\New Data\\Simulation Pull")

    # Load the trained model
    model_path = "PINN_model.pth"
    model = load_model(model_path)

    # Create the scene
    scene = canvas(title="N-Body Simulation (Model Predicted)", width=800, height=600)
    scene.autoscale = True
    
    # Create spheres for each body for visualization
    for b in bodies:
        b['obj'] = sphere(
            pos=b['pos'],
            radius=b['radius'],
            color=b['color'],
            texture=b['texture'],
            make_trail=True,
            retain=5000
        )
    
    # Time parameters
    t = 0.0        # Start time
    h = 0.001       # Timestep in years
    simulation_duration = 1.0  # total simulation time in years

    while t < simulation_duration:
        rate(200)  # 200 frames per second

        # Get current state
        current_state = get_state_vector(bodies)

        # Use the model to predict the next state
        # The model should output the next positions and velocities after 1 timestep
        predictions = model(current_state)

        # Update bodies from the predicted next state
        update_bodies_from_state(bodies, predictions)

        # Update the sphere positions in the scene
        for b in bodies:
            b['obj'].pos = b['pos']

        t += h

simulate()

