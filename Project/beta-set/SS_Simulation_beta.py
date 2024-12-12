

import numpy as np
from scipy.constants import G
import os

# Define constants
M_sun = 1.989e30  # Mass of the Sun in kg
AU = 1.496e11  # Astronomical unit in meters
day_to_seconds = 24 * 3600  # Seconds in a day
G_scaled = G / AU**3 * M_sun * day_to_seconds**2  # Gravitational constant scaled for AU, M_sun, and days

data_directory = 'C:/Users/Manny Admin/Desktop/data/simulations'
os.makedirs(data_directory, exist_ok=True)

# Initialize celestial bodies
bodies = {
    "Mercury": {
        "mass": 3.3011e23 / M_sun,
        "position": np.array([0.39, 0.0, 0.0]),
    },
    "Venus": {
        "mass": 4.8675e24 / M_sun,
        "position": np.array([0.72, 0.0, 0.0]),
    },
    "Earth": {
        "mass": 5.97237e24 / M_sun,
        "position": np.array([1.0, 0.0, 0.0]),
    },
    "Mars": {
        "mass": 6.4171e23 / M_sun,
        "position": np.array([1.52, 0.0, 0.0]),
    },
    "Sun": {
        "mass": 1.0,
        "position": np.array([0.0, 0.0, 0.0]),
    },
}

def orbital_velocity(mass_central, position):
    distance = np.linalg.norm(position)
    velocity_magnitude = np.sqrt(G_scaled * mass_central / distance)
    velocity_direction = np.cross([0, 0, 1], position)
    velocity_direction /= np.linalg.norm(velocity_direction)
    return velocity_magnitude * velocity_direction

# Assign initial velocities
for body, data in bodies.items():
    if body != "Sun":
        data["velocity"] = orbital_velocity(bodies["Sun"]["mass"], data["position"])
    else:
        data["velocity"] = np.zeros(3)

# Simulation parameters
time_step = 1.0  # Time step in days
total_days = 365.25  # Simulate one year
num_steps = int(total_days / time_step)

positions = {body: [data["position"]] for body, data in bodies.items()}
velocities = {body: [data["velocity"]] for body, data in bodies.items()}

# Compute accelerations
def compute_accelerations(positions, masses):
    accelerations = {body: np.zeros(3) for body in positions}
    for body1, pos1 in positions.items():
        for body2, pos2 in positions.items():
            if body1 != body2:
                r_vec = pos2 - pos1
                r_mag = np.linalg.norm(r_vec)
                if r_mag != 0:
                    accelerations[body1] += G_scaled * masses[body2] * r_vec / r_mag**3
    return accelerations

masses = {body: data["mass"] for body, data in bodies.items()}

# Run simulation
for step in range(num_steps):
    current_positions = {body: positions[body][-1] for body in bodies}
    current_velocities = {body: velocities[body][-1] for body in bodies}

    accelerations = compute_accelerations(current_positions, masses)

    for body in bodies:
        half_step_velocity = current_velocities[body] + 0.5 * accelerations[body] * time_step
        new_position = current_positions[body] + half_step_velocity * time_step
        positions[body].append(new_position)

    updated_positions = {body: positions[body][-1] for body in bodies}
    new_accelerations = compute_accelerations(updated_positions, masses)

    for body in bodies:
        new_velocity = current_velocities[body] + 0.5 * (accelerations[body] + new_accelerations[body]) * time_step
        velocities[body].append(new_velocity)

# Save positions and velocities
for body in bodies:
    np.save(os.path.join(data_directory, f"{body}_positions.npy"), np.array(positions[body]))
    np.save(os.path.join(data_directory, f"{body}_velocities.npy"), np.array(velocities[body]))

print("Simulation complete. Data saved.")
