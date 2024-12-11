
import numpy as np
from astroquery.jplhorizons import Horizons
import os

data_directory = 'C:/Users/Manny Admin/Desktop/data/simulations'

bodies = {
    "Mercury": 199,
    "Venus": 299,
    "Earth": 399,
    "Mars": 499,
    "Sun": 10
}

def fetch_nasa_data(bodies, start_date="2023-01-01", end_date="2023-12-31", step="1d"):
    positions = {}
    velocities = {}

    for body, id in bodies.items():
        print(f"Fetching data for {body}...")
        obj = Horizons(id=id, location="500@0", epochs={"start": start_date, "stop": end_date, "step": step})
        eph = obj.vectors()

        pos = np.array([eph["x"], eph["y"], eph["z"]]).T  # AU
        vel = np.array([eph["vx"], eph["vy"], eph["vz"]]).T  # AU/day

        positions[body] = pos
        velocities[body] = vel

    return positions, velocities

def save_nasa_data(positions, velocities):
    os.makedirs(data_directory, exist_ok=True)
    for body, pos in positions.items():
        np.save(os.path.join(data_directory, f"{body}_positions_fetched.npy"), pos)
    for body, vel in velocities.items():
        np.save(os.path.join(data_directory, f"{body}_velocities_fetched.npy"), vel)

def validate_nasa_data(positions, velocities):
    for body, pos in positions.items():
        if np.isnan(pos).any() or np.isinf(pos).any():
            print(f"Invalid positions for {body}")
    for body, vel in velocities.items():
        if np.isnan(vel).any() or np.isinf(vel).any():
            print(f"Invalid velocities for {body}")

if __name__ == "__main__":
    positions, velocities = fetch_nasa_data(bodies)
    validate_nasa_data(positions, velocities)
    save_nasa_data(positions, velocities)
    print("NASA data fetched and saved successfully.")
