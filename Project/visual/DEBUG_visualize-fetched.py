
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Directory containing fetched data
data_directory = 'C:/Users/Manny Admin/Desktop/data/simulations'

# Celestial bodies to analyze
bodies = ["Mercury", "Venus", "Earth", "Mars", "Sun"]

# Load fetched data
def load_fetched_data():
    fetched_data = {}
    for body in bodies:
        positions_file = f"{data_directory}/{body}_positions_fetched.npy"
        if os.path.exists(positions_file):
            positions = np.load(positions_file)
            fetched_data[body] = positions
        else:
            print(f"Warning: {positions_file} not found.")
    return fetched_data

# Visualize fetched orbits
def visualize_fetched_orbits(fetched_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Fetched Orbits")
    ax.set_xlabel("X (AU)")
    ax.set_ylabel("Y (AU)")
    ax.set_zlabel("Z (AU)")

    for body, positions in fetched_data.items():
        positions = np.array(positions)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=body)

    ax.legend()
    plt.show()

if __name__ == "__main__":
    fetched_data = load_fetched_data()
    visualize_fetched_orbits(fetched_data)
