import numpy as np
import matplotlib.pyplot as plt

# Generate recursive spaghetti wave
L = 20            # Length of spaghetti
x = np.linspace(0, L, 1000)
y = np.sin(5 * x) + 0.5 * np.sin(13 * x)  # Recursive harmonics

# Observer cross-section plane (slice at fixed x)
observer_position = int(len(x)/2)  # Take central slice
y_slice = y[observer_position]

# Plot
plt.figure(figsize=(12, 5))

# 1. Full wave structure (spaghetti)
plt.subplot(1, 2, 1)
plt.plot(x, y, label='Recursive Wave (Spaghetti)', linewidth=2)
plt.axvline(x=x[observer_position], color='red', linestyle='--', label='Observer Slice')
plt.title('Recursive Field Structure')
plt.xlabel('Field Axis (X)')
plt.ylabel('Amplitude (Y)')
plt.legend()

# 2. Transversal view seen by observer
plt.subplot(1, 2, 2)
plt.scatter(0, y_slice, color='blue', s=100)
plt.title('Observer View (Transversal Slice)')
plt.xlabel('Field Projection')
plt.ylabel('Amplitude (Projected Y)')
plt.grid(True)

plt.tight_layout()
plt.show()
