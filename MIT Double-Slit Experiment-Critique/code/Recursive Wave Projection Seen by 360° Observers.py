import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
num_angles = 360  # Number of angular observation points
wave_amplitude = 1
wave_frequency = 3
wave_length = 2 * np.pi

# Observer setup
observer_radius = 2  # Radius of observation shell
theta = np.linspace(0, 2 * np.pi, num_angles)
observer_x = observer_radius * np.cos(theta)
observer_y = observer_radius * np.sin(theta)

# Simulated recursive wave structure (spaghetti model)
def recursive_wave(angle, t):
    phase = wave_frequency * angle + t
    return wave_amplitude * np.sin(phase)

# Set up the plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.set_title("Recursive Wave Projection Seen by 360° Observers")

# Plot elements
field_source = plt.Circle((0, 0), 0.05, color='black')
ax.add_artist(field_source)
observers = ax.plot(observer_x, observer_y, 'o', color='gray', alpha=0.3)[0]
projection_dots, = ax.plot([], [], 'o', color='violet')

# Animation function
def update(frame):
    t = frame * 0.1
    projections_x = observer_radius * np.cos(theta)
    projections_y = observer_radius * np.sin(theta)
    intensities = recursive_wave(theta, t)
    colors = plt.cm.plasma((intensities + 1) / 2)  # normalize to 0–1
    projection_dots.set_data(projections_x, projections_y)
    projection_dots.set_color(colors)
    return projection_dots,

ani = FuncAnimation(fig, update, frames=100, interval=100, blit=True)
plt.show()
