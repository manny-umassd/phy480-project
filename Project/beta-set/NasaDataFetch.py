

import numpy as np
from astroquery.jplhorizons import Horizons
import os

data_directory = 'C:/Users/Manny Admin/Desktop/data/simulations'  # Directory to save NASA data

# Constants for celestial bodies
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

        # Extract positions and velocities
        pos = np.array([eph["x"], eph["y"], eph["z"]]).T  # AU
        vel = np.array([eph["vx"], eph["vy"], eph["vz"]]).T  # AU/day

        # Convert velocities from AU/day to km/s
        vel = vel * 1.731456e6

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
            print(f"Invalid positions for {body}: {pos}")
    for body, vel in velocities.items():
        if np.isnan(vel).any() or np.isinf(vel).any():
            print(f"Invalid velocities for {body}: {vel}")

if __name__ == "__main__":
    # Fetch data from NASA's Horizons database
    positions, velocities = fetch_nasa_data(bodies)

    # Validate the data
    validate_nasa_data(positions, velocities)

    # Save the data
    save_nasa_data(positions, velocities)
