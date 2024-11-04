'''
import numpy as np
import matplotlib.pyplot as plt

# Parameters
initial_radius = 1e-3  # initial radius in meters (50 micrometers)
evaporation_constant = 1e-11  # evaporation constant in m^2/s (example value)
time_total = 10  # total time in seconds for the simulation
time_steps = 1000  # number of time steps

# Time array
time = np.linspace(0, time_total, time_steps)

# Calculate radius over time using the D^2 law
radius_squared = initial_radius**2 - evaporation_constant * time
radius = np.sqrt(np.maximum(radius_squared, 0))  # Ensure radius doesn't go negative

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time, radius * 1e6)  # Convert radius to micrometers for plotting
plt.xlabel("Time (s)")
plt.ylabel("Radius (micrometers)")
plt.title("Change in Radius of Evaporating Water Droplet")
plt.grid(True)
plt.show()
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
initial_radius = 50e-6  # Initial radius in meters (50 micrometers)
evaporation_constant = 1e-9  # Evaporation constant in m^2/s
time_total = 3  # Total time in seconds for the animation
fps = 30  # Frames per second for the animation
time_steps = int(time_total * fps)  # Total number of frames

rho_water = 1000
rho_air = 1.225
C_d = 0.47

initial_velocity = 10.0

c_p =  4184
L =  2.26e6
T_air = 293.15 # 20 C
initial_temp = 293.15
h = 10


# Time array
time = np.linspace(0, time_total, time_steps)

# Calculate radius over time using the D^2 law
radius_squared = initial_radius**2 - evaporation_constant * time
radius = np.sqrt(np.maximum(radius_squared, 0))  # Ensure radius doesn't go negative

#Numerical calculation for dv/dt
dt = 0.01
steps = int(time_total/dt)
time_num = np.linspace(0, time_total, steps)

radius_vec = np.zeros(steps)
velocity_vec = np.zeros(steps)
temperature_vec = np.zeros(steps)

radius_vec[0] = initial_radius
velocity_vec[0] = initial_velocity
temperature_vec[0] = initial_temp

for i in range(1, time_steps):
    radius_vec[i] = np.sqrt(max(radius_vec[0]**2 - evaporation_constant * time[i], 0))

    # Update mass
    mass = rho_water * (4/3) * np.pi * radius_vec[i]**3

    area = 4 * np.pi * radius[i]**2
    m_evap_dot = -rho_water * (2 * np.pi * evaporation_constant)
    Q_conv = h * area * (T_air - temperature_vec[i - 1])
    Q_evap = m_evap_dot * L
    dT_dt = (Q_conv - Q_evap) / (mass * c_p)
    
    # Compute drag force
    drag_force = 0.5 * C_d * rho_air * np.pi * radius_vec[i]**2 * velocity_vec[i-1]**2
    
    # Update velocity using dv/dt = -F_d / m
    acceleration = -drag_force / mass
    velocity_vec[i] = velocity_vec[i-1] + acceleration * dt
    temperature_vec[i] = min(temperature_vec[i - 1] + dT_dt*dt, 273.15 + 100)

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-initial_radius * 1e6, initial_radius * 1e6)
ax.set_ylim(-initial_radius * 1e6, initial_radius * 1e6)
ax.set_xlabel("X (micrometers)")
ax.set_ylabel("Y (micrometers)")
plt.title("Evaporating Water Droplet")

# Create the initial circle
circle = plt.Circle((0, 0), initial_radius * 1e6, color='blue', fill=True)
ax.add_patch(circle)

# Update function for animation
def update(frame):
    current_radius = radius[frame] * 1e6  # Convert radius to micrometers
    circle.set_radius(current_radius)
    return circle,

# Create animation
ani = FuncAnimation(fig, update, frames=range(time_steps), blit=True, interval=1000 / fps)


plt.figure(figsize=(10, 6))
plt.plot(time, radius * 1e6)  # Convert radius to micrometers for plotting
plt.plot(time_num, velocity_vec, label="Velocity")
#plt.plot(time_num, temperature_vec, label="???")
plt.xlabel("Time (s)")
plt.ylabel("Radius (micrometers)")
plt.title("Change in Radius of Evaporating Water Droplet")
plt.grid(True)
plt.show()

plt.show()

