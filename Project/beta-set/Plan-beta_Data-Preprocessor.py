import numpy as np
import os

# Add epsilon to avoid division by zero
epsilon = 1e-6

data_directory = 'C:/Users/Manny Admin/Desktop/data/simulations'
preprocessed_data_path = os.path.join(data_directory, 'preprocessed_data.npz')

# Load raw data 
celestial_bodies = ['Mercury', 'Venus', 'Earth', 'Mars', 'Sun']
positions = []
velocities = []
masses = {
    'Mercury': 3.3011e23,
    'Venus': 4.8675e24,
    'Earth': 5.972e24,
    'Mars': 6.4171e23,
    'Sun': 1.989e30
}

for body in celestial_bodies:
    pos = np.load(os.path.join(data_directory, f'{body}_positions.npy'))
    vel = np.load(os.path.join(data_directory, f'{body}_velocities.npy'))
    mass = masses[body]
    positions.append(pos)
    velocities.append(vel)

# Merge positions and velocities
positions = np.vstack(positions)
velocities = np.vstack(velocities)

# Compute normalization stats
mean_pos = np.mean(positions, axis=0)
std_pos = np.std(positions, axis=0)
mean_vel = np.mean(velocities, axis=0)
std_vel = np.std(velocities, axis=0)
global_scale = 1e11  # Scaling factor for positions and velocities

# Precompute features and targets
features = []
targets = []
for body in celestial_bodies:
    pos = np.load(os.path.join(data_directory, f'{body}_positions.npy'))
    vel = np.load(os.path.join(data_directory, f'{body}_velocities.npy'))
    mass = masses[body]

    # Normalize positions and velocities


    pos_norm = (pos - mean_pos) / (std_pos + epsilon)
    vel_norm = (vel - mean_vel) / (std_vel + epsilon)

    mass_norm = (mass - np.mean(list(masses.values()))) / np.std(list(masses.values()))

    # Create features and targets
    for t in range(len(pos) - 1):
        feature = np.hstack((pos_norm[t], vel_norm[t], [mass_norm]))
        target = pos_norm[t + 1]
        features.append(feature)
        targets.append(target)

features = np.array(features, dtype=np.float32)
targets = np.array(targets, dtype=np.float32)

# Save preprocessed data and normalization stats
np.savez(preprocessed_data_path,
         features=features,
         targets=targets,
         mean_pos=mean_pos,
         std_pos=std_pos,
         mean_vel=mean_vel,
         std_vel=std_vel,
         global_scale=global_scale)

print("Preprocessed data saved successfully.")
