
import numpy as np
import os

data_directory = 'C:/Users/Manny Admin/Desktop/data/simulations'

M_sun = 1.989e30
def to_solar_masses(m_kg):
    return m_kg / M_sun

km_s_to_AU_day = 0.0057755

inner_planets = {
    "Mercury": {
        "position": np.array([0.39, 0.0, 0.0]),
        "velocity": np.array([0.0, 47.87 * km_s_to_AU_day, 0.0]),
        "mass": to_solar_masses(3.3011e23)
    },
    "Venus": {
        "position": np.array([0.72, 0.0, 0.0]),
        "velocity": np.array([0.0, 35.02 * km_s_to_AU_day, 0.0]),
        "mass": to_solar_masses(4.8675e24)
    },
    "Earth": {
        "position": np.array([1.0, 0.0, 0.0]),
        "velocity": np.array([0.0, 29.78 * km_s_to_AU_day, 0.0]),
        "mass": to_solar_masses(5.972e24)
    },
    "Mars": {
        "position": np.array([1.52, 0.0, 0.0]),
        "velocity": np.array([0.0, 24.07 * km_s_to_AU_day, 0.0]),
        "mass": to_solar_masses(6.4171e23)
    },
    "Sun": {
        "position": np.array([0.0, 0.0, 0.0]),
        "velocity": np.array([0.0, 0.0, 0.0]),
        "mass": to_solar_masses(1.989e30) # 1 M_sun
    }
}

G = 0.0002959122  # AU^3/(M_sun*day^2)

def simulate_orbits(bodies, num_steps=5000, dt=0.01):
    positions = {body: [data["position"]] for body, data in bodies.items()}
    velocities = {body: [data["velocity"]] for body, data in bodies.items()}

    for step in range(num_steps):
        new_positions = {}
        new_velocities = {}

        for body, data in bodies.items():
            acceleration = np.zeros(3)
            for other_body, other_data in bodies.items():
                if body != other_body:
                    distance_vector = positions[other_body][-1] - positions[body][-1]
                    distance = np.linalg.norm(distance_vector)
                    if distance < 1e-12:
                        distance = 1e-12
                    a = G * other_data["mass"] / (distance ** 2)
                    acceleration += a * (distance_vector / distance)

            new_velocity = velocities[body][-1] + acceleration * dt
            new_position = positions[body][-1] + new_velocity * dt

            new_positions[body] = new_position
            new_velocities[body] = new_velocity

        for body in bodies.keys():
            positions[body].append(new_positions[body])
            velocities[body].append(new_velocities[body])

    return positions, velocities

def save_simulation_data(positions, velocities):
    os.makedirs(data_directory, exist_ok=True)
    for body, body_positions in positions.items():
        np.save(os.path.join(data_directory, f"{body}_positions.npy"), np.array(body_positions))
    for body, body_vels in velocities.items():
        np.save(os.path.join(data_directory, f"{body}_velocities.npy"), np.array(body_vels))

def validate_simulation_data(positions, velocities):
    for body, body_positions in positions.items():
        if np.isnan(body_positions).any() or np.isinf(body_positions).any():
            print(f"Invalid positions for {body}")
    for body, body_vels in velocities.items():
        if np.isnan(body_vels).any() or np.isinf(body_vels).any():
            print(f"Invalid velocities for {body}")

if __name__ == "__main__":
    positions, velocities = simulate_orbits(inner_planets, num_steps=5000, dt=0.01)
    validate_simulation_data(positions, velocities)
    save_simulation_data(positions, velocities)
    print("Simulation data saved successfully.")
