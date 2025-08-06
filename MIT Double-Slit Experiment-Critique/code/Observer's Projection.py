import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3DCOM Recursive Field Simulator
QDF = 0.809728  # Quantum Dimensional Factor
n_layers = 5     # Recursive depth

theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
THETA, PHI = np.meshgrid(theta, phi)

# Harmonic wave with recursive phase
def harmonic_field(n, theta, phi):
    phase = n * (np.sin(theta) + QDF * phi)
    return np.exp(1j * phase) * np.exp(-n / 2)  # Decay with depth

# Qualia Operator (Q^): Projects into observer's cone
def qualia_project(wave):
    return np.abs(wave)**2  # Intensity collapse

# Superposition of recursive waves
field = np.zeros_like(THETA, dtype=complex)
for n in range(1, n_layers + 1):
    field += harmonic_field(n, THETA, PHI)

# Observer's view at θ_obs = π/4 (45°)
θ_obs = np.pi/4
observer_slice = qualia_project(field * np.exp(1j * np.cos(THETA - θ_obs)))

# Plot 3D harmonic field vs. observer's slice
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(THETA, PHI, np.abs(field), cmap='magma')
ax1.set_title("Recursive Field (3DCOM)")

ax2 = fig.add_subplot(122)
ax2.pcolormesh(phi, theta, observer_slice, shading='auto')
ax2.set_title(f"Observer's Projection (θ={θ_obs:.2f})")
plt.show()