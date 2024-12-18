import numpy as np
from vpython import vector, mag, sphere, canvas, rate, textures, color  #Replaces visual module, used to generate the scene and for vector functions
import csv  #Used for saving data in a spreadsheet
import os   #Used for file system functions


G = 4 * (np.pi**2)  # Gravitational constant in AU^3 / (Solar Mass) / (Year^2)

bodies = [
   {
        'name': 'Sun',
        'mass': 1.0,  #In solar masses
        'pos': vector(0, 0, 0),  #Initial position, in AU
        'vel': vector(0, 0, 0),  #Initial velocity, in AU/yr
        'radius': 0.2,           #Size of sphere, visual purposes only
        'color': color.yellow,
        'texture': None,         #Because I wanted Earth to have the Earth texture, all bodies must have this line set to 'none' for how the bodies are called in later. 
    },
    {
        'name': 'Earth',
        'mass': 3.0e-6,  #In Solar masses
        'pos': vector(1.0, 0, 0),  # Distance ~1 AU from the Sun
        'vel': vector(0, 0, -6.179),  # Pulled from table in cpms-ch04
        'radius': 0.1,             #Visual purposes only
        'color': color.white,      #For blank sphere so texture can be the sphere's "color"
        'texture': textures.earth, #home
    },
    {
        'name': 'Jupiter',   #Chosen for it's large mass. Helps cause of perturbation on the Sun
        'mass': 9.5e-4,      #In solar masses. All masses are pulled from the table in cpms-ch04
        'pos': vector(5.2, 0, 0),  # Distance ~5.2 AU from the Sun
        'vel': vector(0, 0, -2.624),  # Pulled from table in cpms-ch04
        'radius': 0.15,
        'color': color.orange,
        'texture': None,
    },
]

def compute_accelerations(bodies):
    """
    Compute the gravitational acceleration on each body due to all others.

    Parameters:
    bodies (list of dict): List of bodies with 'pos', 'mass' keys at least.

    Returns:
    list of vector: A list of accelerations corresponding to each body.
    """
    # Initialize a list of accelerations with zero vectors for all mentioned bodies. (Can include others)
    accels = [vector(0,0,0) for _ in range(len(bodies))]
    
    # For each pair of bodies i, j, compute gravitational acceleration contribution
    for i in range(len(bodies)):
        for j in range(len(bodies)):
            if i != j: #loop works as long as the two bodies referenced aren't the same. 
                # Relative position vector from body j to i
                rij = bodies[i]['pos'] - bodies[j]['pos']
                dist = mag(rij)
                # Acceleration contribution from body j on i, added to list in location for specified body.
                accels[i] += -G * bodies[j]['mass'] * rij / (dist**3)
    return accels

def save_data_incremental(t, bodies, filename="C:\\Users\\Manny Admin\\Desktop\\New Data\\Simulation Pull\\simulation_data.csv"):
     
    if not os.path.exists(filename):
        # Write the header if the file doesn't exist
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            headers = ["time"]
            for body in bodies:
                #Creates the rest of the header for the .csv file (e.g Sun_x, Mercury_vz) for all bodies in order listed above.
                headers.extend([f"{body['name']}_x", f"{body['name']}_y", f"{body['name']}_z",
                                f"{body['name']}_vx", f"{body['name']}_vy", f"{body['name']}_vz"])
            writer.writerow(headers)

    # Append the data row to the csv file
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        row = [t]
        for body in bodies:
            row.extend([body['pos'].x, body['pos'].y, body['pos'].z,
                        body['vel'].x, body['vel'].y, body['vel'].z])
        writer.writerow(row)

def simulate():

    # Run the N-body simulation using a Leapfrog integrator and visualize it.
 
    # Create the scene
    scene = canvas(title="N-Body Simulation", width=800, height=600)
    scene.autoscale = True
    
    # Create spheres for each body for visualization
    for b in bodies:
        b['obj'] = sphere(
            pos=b['pos'],
            radius=b['radius'],
            color=b['color'],
            texture=b['texture'],
            make_trail=True,
            retain=5000  # Keep a long but finite trail
        )
    
    # Time parameters
    t = 0.0        # Start time
    h = 0.001       # Timestep in years
    simulation_duration = 10.0  # Total simulation duration in years
    data = []       # Storage for simulation data
    
    # Initial accelerations
    accels = compute_accelerations(bodies)
    
    # Compute half-step velocities for leapfrog integrator
    for i, b in enumerate(bodies):
        # v_half = v + 0.5 * a * h
        b['v_half'] = b['vel'] + 0.5 * accels[i] * h
    
    # Main simulation loop
    while t < simulation_duration:
        rate(200)  # Limit to 200 frames per second
        
        # Save current timestep data incrementally
        save_data_incremental(t, bodies)
        
        # Update positions of all bodies using v_half
        for i, b in enumerate(bodies):
            # new_r = r + v_half * h
            b['pos'] = b['pos'] + b['v_half'] * h
        
        # Compute new accelerations after moving bodies
        accels = compute_accelerations(bodies)
        
        # Update velocities for all bodies
        for i, b in enumerate(bodies):
            # new_v_half = v_half + a * h
            b['v_half'] = b['v_half'] + accels[i] * h
            
            # Update the sphere positions in the scene
            b['obj'].pos = b['pos']
        
        t += h
        
simulate()