import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D

# GNN Model Definition 
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc_out(x)
        return x

# Load the trained model
data_directory = 'C:/Users/Manny Admin/Desktop/data/simulations'
model_path = f'{data_directory}/trained_gnn_model.pth'
preprocessed_data = np.load(f'{data_directory}/preprocessed_data.npz')

# Load normalization stats
norm_stats = {
    'mean_pos': preprocessed_data['mean_pos'],
    'std_pos': preprocessed_data['std_pos'],
    'mean_vel': preprocessed_data['mean_vel'],
    'std_vel': preprocessed_data['std_vel'],
    'global_scale': preprocessed_data['global_scale']
}

# Add epsilon to avoid division by zero
epsilon = 1e-8
norm_stats['std_pos'] = np.maximum(norm_stats['std_pos'], epsilon)
norm_stats['std_vel'] = np.maximum(norm_stats['std_vel'], epsilon)

# Define masses for each body (constants)
body_masses = {
    'Mercury': 0.330,
    'Venus': 4.87,
    'Earth': 5.97,
    'Mars': 0.642,
    'Sun': 1989000  # Masses are set relative to 10^24 kg (Earth ~5.97 * 10^24 kg)
}

# Initialize the model
input_dim = preprocessed_data['features'].shape[1]
output_dim = preprocessed_data['targets'].shape[1]
hidden_dim = 128
model = GNNModel(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load(model_path))
model.eval()

# Predict Orbits
def predict_orbits(model, norm_stats, steps=365):
    bodies = ['Mercury', 'Venus', 'Earth', 'Mars', 'Sun']
    positions = {body: [] for body in bodies}

    # Load initial data
    initial_positions = {body: np.load(f'{data_directory}/{body}_positions.npy')[0] for body in bodies}
    initial_velocities = {body: np.load(f'{data_directory}/{body}_velocities.npy')[0] for body in bodies}

    # Validate initial data
    for body in bodies:
        initial_positions[body] = np.nan_to_num(initial_positions[body])
        initial_velocities[body] = np.nan_to_num(initial_velocities[body])

    global_scale = norm_stats['global_scale']
    current_features = []

    # Normalize initial features (but Sun)
    for body in bodies:
        if body == 'Sun':
            positions['Sun'].append(initial_positions['Sun'])  # Fix Sun's position
            continue

        pos = initial_positions[body] / global_scale
        vel = initial_velocities[body] / global_scale
        mass = body_masses[body] / global_scale  # Use constant mass values
        node_feat = np.hstack(((pos - norm_stats['mean_pos']) / norm_stats['std_pos'],
                                (vel - norm_stats['mean_vel']) / norm_stats['std_vel'],
                                [mass]))  # Add mass as a feature
        current_features.append(node_feat)

    current_features = torch.tensor(current_features, dtype=torch.float32)

    for step in range(steps):
        with torch.no_grad():
            predicted_positions = model(current_features).numpy()
            predicted_positions = np.clip(predicted_positions, -1e5, 1e5)  # Clamp values
        print(f"Step {step}: {predicted_positions}")  # Debug

        feature_index = 0
        for body in bodies:
            if body == 'Sun':
                positions['Sun'].append(initial_positions['Sun'])  # Keep Sun static
                continue

            pos_denorm = predicted_positions[feature_index] * norm_stats['std_pos'] + norm_stats['mean_pos']
            positions[body].append(pos_denorm)

            # Update features for the next prediction
            current_features[feature_index, :3] = torch.tensor(predicted_positions[feature_index], dtype=torch.float32)
            feature_index += 1

    print("Final Predicted Positions:", positions)  # Debug
    return positions

# Visualize Predicted Orbits
def visualize_predicted_orbits(predicted_positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Predicted Orbits")
    ax.set_xlabel("X (AU)")
    ax.set_ylabel("Y (AU)")
    ax.set_zlabel("Z (AU)")

    for body, body_positions in predicted_positions.items():
        body_positions = np.array(body_positions)
        print(f"{body} positions range: {body_positions.min(axis=0)} to {body_positions.max(axis=0)}")  # Debug
        ax.plot(body_positions[:, 0], body_positions[:, 1], body_positions[:, 2], label=body)

    ax.legend()
    plt.show()

# Main
if __name__ == "__main__":
    predicted_positions = predict_orbits(model, norm_stats)
    visualize_predicted_orbits(predicted_positions)
