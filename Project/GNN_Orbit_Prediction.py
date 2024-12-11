import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn as nn
import os
import matplotlib.pyplot as plt

data_directory = 'C:/Users/Manny Admin/Desktop/data/simulations'
model_path = os.path.join(data_directory, 'gnn_model.pth')
norm_stats_path = os.path.join(data_directory, 'norm_stats.npz')

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return x

def load_simulation_data():
    celestial_bodies = ['Mercury', 'Venus', 'Earth', 'Mars', 'Sun']
    simulation_data = {}

    for body in celestial_bodies:
        positions = np.load(os.path.join(data_directory, f'{body}_positions.npy'))
        velocities = np.load(os.path.join(data_directory, f'{body}_velocities.npy'))
        simulation_data[body] = {
            'positions': positions,
            'velocities': velocities,
            'masses': np.array([3.3011e23, 4.8675e24, 5.972e24, 6.4171e23, 1.989e30])
        }

    return simulation_data

def create_prediction_graph(celestial_data, norm_stats):
    mean_pos = norm_stats['mean_pos']
    std_pos = norm_stats['std_pos']
    mean_vel = norm_stats['mean_vel']
    std_vel = norm_stats['std_vel']
    global_scale = norm_stats['global_scale']

    celestial_bodies = list(celestial_data.keys())
    masses = np.array([3.3011e23, 4.8675e24, 5.972e24, 6.4171e23, 1.989e30])
    nodes = []

    for i, body in enumerate(celestial_bodies):
        pos = celestial_data[body]['positions'][-1]
        vel = celestial_data[body]['velocities'][-1]

        # Apply same global scaling as in training
        pos_scaled = pos / global_scale
        vel_scaled = vel / global_scale

        mass = masses[i]
        nodes.append(np.hstack((pos_scaled, vel_scaled, [mass])))

    nodes = np.array(nodes, dtype=np.float32)
    # Normalize as done in training
    nodes[:, :3] = (nodes[:, :3] - mean_pos) / (std_pos + 1e-6)
    nodes[:, 3:6] = (nodes[:, 3:6] - mean_vel) / (std_vel + 1e-6)

    # Edges
    num_nodes = len(nodes)
    edges = []
    edge_attrs = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append((i, j))
                dist = np.linalg.norm(nodes[i][:3] - nodes[j][:3])
                dist = max(dist, 1e-6)
                edge_attrs.append([1/dist])

    edge_attrs = np.array(edge_attrs)
    ea_min, ea_max = np.min(edge_attrs), np.max(edge_attrs)
    edge_attrs = (edge_attrs - ea_min)/(ea_max - ea_min + 1e-6)

    x = torch.tensor(nodes, dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), celestial_bodies

def predict_orbits(model, celestial_data, steps=100):
    model.eval()
    stats = np.load(os.path.join(data_directory, 'norm_stats.npz'))
    mean_pos = stats['mean_pos']
    std_pos = stats['std_pos']
    mean_vel = stats['mean_vel']
    std_vel = stats['std_vel']
    global_scale = stats['global_scale']

    graph, celestial_bodies = create_prediction_graph(celestial_data, stats)
    x, edge_index = graph.x, graph.edge_index

    predictions = []
    for step in range(steps):
        with torch.no_grad():
            pred_norm = model(x, edge_index)
        pred_norm_np = pred_norm.cpu().numpy()

        # De-normalize predictions
        pred_phys = (pred_norm_np * (std_pos + 1e-6)) + mean_pos
        pred_phys = pred_phys * global_scale  # revert global scaling

        predictions.append(pred_phys)

        # Re-normalize before feeding back
        pred_norm_back = (pred_phys/global_scale - mean_pos)/(std_pos + 1e-6)
        x[:, :3] = torch.tensor(pred_norm_back, dtype=torch.float32)

    return np.array(predictions), celestial_bodies

def plot_orbits(predicted_orbits, celestial_bodies):
    plt.figure(figsize=(10, 10))
    for i, body in enumerate(celestial_bodies):
        orbit = predicted_orbits[:, i, :]
        plt.plot(orbit[:, 0], orbit[:, 1], label=body)
    plt.xlabel('X Position (AU)')
    plt.ylabel('Y Position (AU)')
    plt.title('Predicted Orbits of Inner Planets')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    # Load trained model
    gnn_model = GNNModel(input_dim=7, hidden_dim=128, output_dim=3)
    gnn_model.load_state_dict(torch.load(model_path))

    # Load simulation data
    simulation_data = load_simulation_data()

    # Predict orbits
    predicted_orbits, celestial_bodies = predict_orbits(gnn_model, simulation_data, steps=100)

    # Save predictions
    np.save(os.path.join(data_directory, 'predicted_orbits.npy'), predicted_orbits)
    print("Predicted orbits saved successfully.")

    # Plot orbits
    plot_orbits(predicted_orbits, celestial_bodies)
