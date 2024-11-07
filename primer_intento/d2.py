import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#
mu, sigma = 50, 10

# Parameters
initial_radius   = np.random.normal(mu, sigma)*10**(-6)
initial_velocity = np.random.normal(10, 3)

time_total = 12                     # Total time in seconds for the animation
fps = 120                           # Frames per second for the animation
time_steps = int(time_total * fps) # Total number of frames
evaporation_constant = 1e-9        # Evaporation constant in m^2/s
# Constantes fisica
rho_water    = 1000   # Densidad del agua kg/m^3
rho_air      = 1.225  # Densidad del aire kg/m^3
C_d          = 0.47   #
c_p          = 4184   # Calor especifico del agua
L            = 2.26e6 #
T_air        = 293.15 # 20 C
initial_temp = 293.15
h            = 10


def simular_gota(time, radio_inicial, velocidad_inicial):
    radio_cuadrado = radio_inicial**2 - evaporation_constant * time # Aplicar ley D^2
    radio          = np.sqrt(np.maximum(radio_cuadrado, 0))         # Cortar si el radio se va a negativo

    # Ahora hacemos calculo numerico, en particular implementamos el metodo de euler para
    # obtener la velocidad
    #
    dt = time_total/time_steps
    print(dt)

    masa      = rho_water*(4/3)*np.pi*(radio**3)
    velocidad = np.zeros(time_steps)

    velocidad[0] = velocidad_inicial
    for i in range(1, time_steps):
        drag_force   = 0.5 * C_d * rho_air * np.pi * radio[i]**2 * velocidad[i - 1]**2
        aceleracion  = -drag_force/masa[i]
        velocidad[i] = velocidad[i-1] + aceleracion * dt

    return (radio, velocidad)




'''
# Time array
time = np.linspace(0, time_total, time_steps)

# Calculate radius over time using the D^2 law
radius_squared = initial_radius**2 - evaporation_constant * time
radius = np.sqrt(np.maximum(radius_squared, 0))  # Ensure radius doesn't go negative

#Numerical calculation for dv/dt dt = 0.01
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
'''
time = np.linspace(0, time_total, time_steps)

(radius, velocity_vec) = simular_gota(time, initial_radius, initial_velocity)

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


#plt.figure(figsize=(10, 6))
for i in range(100):
    initial_radius   = np.random.normal(mu, sigma)*10**(-6)
    initial_velocity = np.random.normal(10, 3)
    (radius, velocity) = simular_gota(time, initial_radius, initial_velocity)
    plt.figure(100)
    plt.plot(time, radius * 1e6)  # Convert radius to micrometers for plotting
    plt.figure(200)
    plt.plot(time, velocity)

plt.figure(100)
plt.xlabel("Tiempo (s)")
plt.ylabel("Radios (micrometros)")
plt.title("Radio (Radio Inicial={})".format(initial_radius))
plt.grid(True)
plt.show()

plt.figure(200)
plt.figure(figsize=(10, 6))
plt.plot(time, velocity_vec, label="Velocity")
plt.xlabel("Time (s)")
plt.ylabel("Velocidad (m/s)")
plt.title("Velocidad")
plt.grid(True)
plt.show()
