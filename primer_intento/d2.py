import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

mu, sigma = 50, 10
mu_v, sigma_v = 10, 3
mu_ang, sigma_ang = 0, 10

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

# Decision de diseno, simularemos la gota en el momento de su creacion, cada gota es independiente de la otra
class Gota:
    def __init__(self, radii = None, vel_inicial = None):
        if radii == None:
            self.initial_radius = np.random.normal(mu, sigma)*10**(-6) #Si no hay radio inicial obtener de una normal en el orden de los 50 micrometros
        else:
            self.initial_radius = radii

        if vel_inicial == None:
            self.initial_vel = np.random.normal(mu_v, sigma_v)
        else:
            self.initial_vel = vel_inicial

        self.angle     = np.random.normal(mu_ang, sigma_ang)*np.pi/180 #Angulo en radianes
        self.velocidad = np.zeros(time_steps)
        self.distancias = np.zeros(time_steps)

        self.time      = np.linspace(0, time_total, time_steps)
        radio_cuadrado = self.initial_radius**2 - evaporation_constant * self.time #Ley D^2
        self.radio     = np.sqrt(np.maximum(radio_cuadrado, 0))

        self.masa = rho_water*(4/3)*np.pi*(self.radio**3)

        dt        = time_total/time_steps

        self.velocidad[0]  = self.initial_vel
        self.distancias[0] = self.velocidad[0] * dt
        for i in range(1, time_steps):
            drag_force  = 0.5 * C_d * rho_air * np.pi * self.radio[i]**2 * self.velocidad[i - 1]**2
            aceleracion = -drag_force/self.masa[i]
            self.velocidad[i] = self.velocidad[i - 1] + aceleracion * dt
            if np.isnan(self.velocidad[i]):
                self.velocidad[i] = 0

            self.distancias[i] = self.distancias[i - 1] + self.velocidad[i] * dt

        self.tiempo_de_desintegracion = np.where(self.radio == 0)[0][0]
        self.distancia_recorrida = self.distancias[-1]


gota_animacion = Gota(50e-6, 10)

'''
# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-gota_animacion.initial_radius * 1e6, gota_animacion.initial_radius * 1e6)
ax.set_ylim(-gota_animacion.initial_radius * 1e6, gota_animacion.initial_radius * 1e6)
ax.set_xlabel("X (micrometers)")
ax.set_ylabel("Y (micrometers)")
plt.title("Evaporating Water Droplet")

# Create the initial circle
circle = plt.Circle((0, 0), gota_animacion.initial_radius * 1e6, color='blue', fill=True)
ax.add_patch(circle)

# Update function for animation
def update(frame):
    current_radius = gota_animacion.radio[frame] * 1e6  # Convert radius to micrometers
    circle.set_radius(current_radius)
    return circle,

# Create animation
ani = FuncAnimation(fig, update, frames=range(time_steps), blit=True, interval=1000 / fps)
'''


#plt.figure(figsize=(10, 6))

gotas      = []
tiempos    = []
distancias = []
radios     = []
angulos    = []
xs = []
ys = []
for i in range(100):
    gota = Gota()

    gotas.append(gota)
    tiempos.append(gota.tiempo_de_desintegracion*(time_total/time_steps))
    distancias.append(gota.distancia_recorrida)
    angulos.append(gota.angle*180/np.pi)

    xs.append(gota.distancia_recorrida*np.cos(gota.angle))
    ys.append(gota.distancia_recorrida*np.sin(gota.angle))

    radios.append(gota.initial_radius)

    plt.figure("Radios")
    plt.plot(gota.time, gota.radio * 1e6)  # Convert radius to micrometers for plotting
    plt.figure("Velocidades")
    plt.plot(gota.time, gota.velocidad)


plt.figure("Distribucion de tiempos de desintegracion")
plt.hist(tiempos, edgecolor='black')

plt.figure("Distribucion de distancias recorridas")
print(distancias)
plt.hist(distancias, edgecolor='black')

plt.figure("Simulacion estornudo")
plt.scatter(xs, ys)

plt.figure("Distribucion de radios iniciales")
plt.hist(radios, edgecolor='black')

plt.figure("Distribucion de angulos")
plt.hist(angulos, edgecolor='black')

positions_x = np.array([gota.distancias * np.cos(gota.angle) for gota in gotas])
positions_y = np.array([gota.distancias * np.sin(gota.angle) for gota in gotas])
radii = np.array([gota.radio * 1e6 for gota in gotas])  # Convert to micrometers for animation

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(0, 2)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel("X (Metros)")
ax.set_ylabel("Y (Metros)")
plt.title("Evaporating Droplets Scatter Animation")

scat = ax.scatter(positions_x[:, 0], positions_y[:, 0], s=radii[:, 0], color='blue', alpha=0.5)

def update(frame):
    current_radii = radii[:, frame]
    scat.set_offsets(np.c_[positions_x[:, frame], positions_y[:, frame]])
    return scat,

ani = FuncAnimation(fig, update, frames=time_steps, blit=True, interval=1000 / fps)
#FFwriter = animation.FFMpegWriter(fps=120)
#ani.save('scatter.mp4', writer=FFwriter)

plt.show()
