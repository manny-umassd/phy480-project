
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Directory containing simulation data
data_directory = 'C:/Users/Manny Admin/Desktop/data/simulations'

# Celestial bodies to analyze
bodies = ["Mercury", "Venus", "Earth", "Mars", "Sun"]

# Load simulated data
def load_simulated_data():
    simulated_data = {}
    for body in bodies:
        positions_file = f"{data_directory}/{body}_positions.npy"
        if os.path.exists(positions_file):
            positions = np.load(positions_file)
            simulated_data[body] = positions
        else:
            print(f"Warning: {positions_file} not found.")
    return simulated_data

# Visualize simulated orbits
def visualize_simulated_orbits(simulated_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Simulated Orbits")
    ax.set_xlabel("X (AU)")
    ax.set_ylabel("Y (AU)")
    ax.set_zlabel("Z (AU)")

    for body, positions in simulated_data.items():
        positions = np.array(positions)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=body)

    ax.legend()
    plt.show()

if __name__ == "__main__":
    simulated_data = load_simulated_data()
    visualize_simulated_orbits(simulated_data)