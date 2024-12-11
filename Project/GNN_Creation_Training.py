from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random

data_directory = 'C:/Users/Manny Admin/Desktop/data/simulations'

class GNNModel(nn.Module):
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

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def body_mass(body):
    masses = {
        'Mercury': 3.3011e23,
        'Venus': 4.8675e24,
        'Earth': 5.972e24,
        'Mars': 6.4171e23,
        'Sun': 1.989e30
    }
    return masses[body]

def load_simulation_and_fetched_data():
    celestial_bodies = ['Mercury', 'Venus', 'Earth', 'Mars', 'Sun']
    simulation_data = {}
    fetched_data = {}

    for body in celestial_bodies:
        sim_positions = np.load(os.path.join(data_directory, f'{body}_positions.npy'))
        sim_velocities = np.load(os.path.join(data_directory, f'{body}_velocities.npy'))
        simulation_data[body] = {
            'positions': sim_positions,
            'velocities': sim_velocities,
            'mass': body_mass(body)
        }

        fetch_positions = np.load(os.path.join(data_directory, f'{body}_positions_fetched.npy'))
        fetch_velocities = np.load(os.path.join(data_directory, f'{body}_velocities_fetched.npy'))
        fetched_data[body] = {
            'positions': fetch_positions,
            'velocities': fetch_velocities,
            'mass': body_mass(body)
        }

    return simulation_data, fetched_data

def compute_norm_stats(simulation_data):
    all_pos = []
    all_vel = []
    all_mass = []
    global_scale = 1e11 # Slight scaling

    for body, bd in simulation_data.items():
        steps = bd['positions'].shape[0]
        sample_indices = np.random.choice(steps, min(1000, steps), replace=False)
        for idx in sample_indices:
            pos = bd['positions'][idx]/global_scale
            vel = bd['velocities'][idx]/global_scale
            all_pos.append(pos)
            all_vel.append(vel)
            all_mass.append(bd['mass'])

    all_pos = np.array(all_pos)
    all_vel = np.array(all_vel)
    all_mass = np.array(all_mass)

    mean_pos = np.mean(all_pos, axis=0)
    std_pos = np.std(all_pos, axis=0); std_pos = np.where(std_pos==0,1e-6,std_pos)
    mean_vel = np.mean(all_vel, axis=0)
    std_vel = np.std(all_vel, axis=0); std_vel = np.where(std_vel==0,1e-6,std_vel)
    mean_mass = np.mean(all_mass)
    std_mass = np.std(all_mass) if np.std(all_mass)!=0 else 1e-6

    norm_stats = {
        'mean_pos': mean_pos,
        'std_pos': std_pos,
        'mean_vel': mean_vel,
        'std_vel': std_vel,
        'mean_mass': mean_mass,
        'std_mass': std_mass,
        'global_scale': global_scale
    }
    return norm_stats

def create_graph(input_bodies, norm_stats, t):
    mean_pos = norm_stats['mean_pos']
    std_pos = norm_stats['std_pos']
    mean_vel = norm_stats['mean_vel']
    std_vel = norm_stats['std_vel']
    mean_mass = norm_stats['mean_mass']
    std_mass = norm_stats['std_mass']
    global_scale = norm_stats['global_scale']

    bodies = list(input_bodies.keys())
    num_nodes = len(bodies)

    nodes = []
    for body in bodies:
        pos = input_bodies[body]['positions'][t]/global_scale
        vel = input_bodies[body]['velocities'][t]/global_scale
        mass = input_bodies[body]['mass']

        pos_norm = (pos - mean_pos)/(std_pos+1e-6)
        vel_norm = (vel - mean_vel)/(std_vel+1e-6)
        mass_norm = (mass - mean_mass)/(std_mass+1e-6)
        node_feat = np.hstack((pos_norm, vel_norm, [mass_norm]))
        nodes.append(node_feat)
    nodes = np.array(nodes, dtype=np.float32)

    edges = []
    edge_attrs = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i!=j:
                edges.append((i,j))
                dist = np.linalg.norm(nodes[i][:3]-nodes[j][:3])
                dist = max(dist,1e-6)
                edge_attrs.append([1/dist])

    edge_attrs = np.array(edge_attrs,dtype=np.float64)
    ea_min,ea_max = np.min(edge_attrs),np.max(edge_attrs)
    edge_attrs=(edge_attrs-ea_min)/(ea_max-ea_min+1e-6)

    x = torch.tensor(nodes,dtype=torch.float32)
    edge_index = torch.tensor(edges,dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs,dtype=torch.float32)

    # Targets- next positions (no noise)
    targets = []
    for body in bodies:
        pos_next = input_bodies[body]['positions'][t+1]/global_scale
        pos_next_norm = (pos_next - mean_pos)/(std_pos+1e-6)
        targets.append(pos_next_norm)

    targets = torch.tensor(targets, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.targets = targets
    return data

def build_dataset(simulation_data, fetched_data, norm_stats, num_samples=2000):
    dataset = []
    def collect_samples(data_source):
        bodies = list(data_source.keys())
        steps_list = [data_source[b]['positions'].shape[0] for b in bodies]
        min_steps = min(steps_list)
        # More samples, random timesteps
        for _ in range(num_samples//2):
            t = random.randint(0, min_steps-2)
            g = create_graph(data_source, norm_stats, t)
            dataset.append(g)

    collect_samples(simulation_data)
    collect_samples(fetched_data)
    return dataset

def train_gnn_model(model, simulation_data, fetched_data, epochs=500, lr=0.0005, weight_decay=1e-5):
    norm_stats = compute_norm_stats(simulation_data)
    np.savez(os.path.join(data_directory, 'norm_stats.npz'), **norm_stats)

    dataset = build_dataset(simulation_data, fetched_data, norm_stats, num_samples=2000)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            inputs = batch.x
            edge_index = batch.edge_index
            targets = batch.targets
            optimizer.zero_grad()
            outputs = model(inputs, edge_index)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch%50==0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4e}, LR: {scheduler.get_last_lr()}")
        scheduler.step()

    torch.save(model.state_dict(), os.path.join(data_directory, 'gnn_model.pth'))
    print("Model training completed and saved.")

if __name__=="__main__":
    simulation_data, fetched_data = load_simulation_and_fetched_data()
    model = GNNModel(input_dim=7, hidden_dim=128, output_dim=3) # Only predicting next positions
    initialize_weights(model)
    train_gnn_model(model, simulation_data, fetched_data, epochs=500, lr=0.0005)
