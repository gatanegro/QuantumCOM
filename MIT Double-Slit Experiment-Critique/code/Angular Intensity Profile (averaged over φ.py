import numpy as np
import matplotlib.pyplot as plt

# Constants from 3DCOM framework
QDF = 0.810058772143807  # Quantum Dimensional Factor
LZ = 1.23498228  # Loop Zero attractor constant

# Harmonic recursion layers to simulate
n_layers = 10

# Discretize observation angles theta (0 to pi), phi (0 to 2pi)
theta = np.linspace(0, np.pi, 180)
phi = np.linspace(0, 2 * np.pi, 360)
THETA, PHI = np.meshgrid(theta, phi)

# Base wave amplitude function for harmonic n
def harmonic_wave(n, theta, phi):
    # Simulate phase oscillation modulated by harmonic level and angle
    phase = n * (theta + phi) * QDF * np.pi
    amplitude = np.cos(phase) * np.exp(-n * 0.1)  # Exponential decay with n
    return amplitude

# Qualia Operator Q^: projects harmonic superposition onto perceptual node
def qualia_projection(wave_sum):
    # Simplified nonlinear mirror: square and normalize
    intensity = wave_sum ** 2
    return intensity / np.max(intensity)

# Compute wave superposition across n_layers
wave_superposition = np.zeros_like(THETA)
for n in range(1, n_layers + 1):
    wave_superposition += harmonic_wave(n, THETA, PHI)

# Project with Qualia Operator Q^
intensity_map = qualia_projection(wave_superposition)

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Qualia Projected Interference Intensity (θ, φ)")
plt.pcolormesh(PHI, THETA, intensity_map, shading='auto', cmap='inferno')
plt.xlabel('φ (rad)')
plt.ylabel('θ (rad)')
plt.colorbar(label='Normalized Intensity')

# Plot intensity as function of θ averaged over φ
avg_intensity_theta = np.mean(intensity_map, axis=0)
plt.subplot(1, 2, 2)
plt.title("Angular Intensity Profile (averaged over φ)")
plt.plot(theta, avg_intensity_theta)
plt.xlabel('θ (rad)')
plt.ylabel('Average Intensity')
plt.tight_layout()
plt.show()