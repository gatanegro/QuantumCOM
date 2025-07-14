import numpy as np
from sympy import isprime
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_3dcom_sequence(n, steps=100):
    """Generate sequence with 3DCOM coordinates (value, prime-gap, octave-phase)"""
    seq = []
    prev_p = 2  # Track previous prime
    for _ in range(steps):
        if n == 1:
            break

        # Track prime gaps
        prime_gap = 0
        if isprime(n):
            prime_gap = n - prev_p
            prev_p = n

        # 3DCOM coordinates
        x = n                            # Raw value
        y = prime_gap                    # Prime gap
        z = (n % 9) * (2*np.pi/9)        # Octave phase

        seq.append((x, y, z))

        # Collatz step
        n = n//2 if n % 2 == 0 else 3*n + 1
    return seq


def coupled_3dcom(n1, n2, steps=100, coupling=0.3):
    """Create coupled 3DCOM sequences with phase synchronization"""
    s1, s2 = [], []
    prev_p1, prev_p2 = 2, 2

    for _ in range(steps):
        if n1 == 1 or n2 == 1:
            break

        # Prime tracking
        pg1, pg2 = 0, 0
        if isprime(n1):
            pg1 = n1 - prev_p1
            prev_p1 = n1
        if isprime(n2):
            pg2 = n2 - prev_p2
            prev_p2 = n2

        # Phase coupling
        phase_diff = ((n1 % 9) - (n2 % 9)) * (2*np.pi/9)
        mod1 = 1 + coupling*np.sin(phase_diff)
        mod2 = 1 + coupling*np.cos(phase_diff)

        # Store 3D coordinates
        s1.append((n1, pg1, (n1 % 9)*2*np.pi/9))
        s2.append((n2, pg2, (n2 % 9)*2*np.pi/9))

        # Coupled evolution
        n1 = int((n1//2 if n1 % 2 == 0 else 3*n1+1) * mod1)
        n2 = int((n2//2 if n2 % 2 == 0 else 3*n2+1) * mod2)

    return s1, s2


def compute_overlap(s1, s2):
    """Quantify prime and phase overlap in 3DCOM space"""
    primes1 = {x[0] for x in s1 if isprime(x[0])}
    primes2 = {x[0] for x in s2 if isprime(x[0])}
    prime_overlap = len(primes1 & primes2) / \
        max(1, min(len(primes1), len(primes2)))

    # Phase synchronization
    min_len = min(len(s1), len(s2))
    phase_sync = np.mean([np.cos(s1[i][2]-s2[i][2]) for i in range(min_len)])

    return (prime_overlap + phase_sync)/2  # Combined metric [0,1]


# Generate systems
np.random.seed(42)
coupled_pairs = [coupled_3dcom(7, 8) for _ in range(50)]
uncoupled_pairs = [(generate_3dcom_sequence(
    7), generate_3dcom_sequence(8)) for _ in range(50)]

# Calculate overlaps
coupled_overlaps = [compute_overlap(s1, s2) for s1, s2 in coupled_pairs]
uncoupled_overlaps = [compute_overlap(s1, s2) for s1, s2 in uncoupled_pairs]

# Statistical test
t_stat, p_value = ttest_ind(coupled_overlaps, uncoupled_overlaps)
effect_size = (np.mean(coupled_overlaps) -
               np.mean(uncoupled_overlaps))/np.std(uncoupled_overlaps)

print(
    f"Coupled Overlap: {np.mean(coupled_overlaps):.3f} ± {np.std(coupled_overlaps):.3f}")
print(
    f"Uncoupled Overlap: {np.mean(uncoupled_overlaps):.3f} ± {np.std(uncoupled_overlaps):.3f}")
print(f"p-value: {p_value:.5f}, Cohen's d: {effect_size:.2f}")

# 3D Visualization
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Plot coupled pair
s1, s2 = coupled_pairs[0]
ax1.scatter([x[0] for x in s1], [x[1] for x in s1], [x[2]
            for x in s1], c='blue', label='n=7')
ax1.scatter([x[0] for x in s2], [x[1] for x in s2], [x[2]
            for x in s2], c='red', label='n=8')
ax1.set_title("Coupled 3DCOM Trajectories")
ax1.set_xlabel("Value")
ax1.set_ylabel("Prime Gap")
ax1.set_zlabel("Phase")

# Plot uncoupled pair
s1, s2 = uncoupled_pairs[0]
ax2.scatter([x[0] for x in s1], [x[1] for x in s1], [x[2]
            for x in s1], c='blue', label='n=7')
ax2.scatter([x[0] for x in s2], [x[1] for x in s2], [x[2]
            for x in s2], c='red', label='n=8')
ax2.set_title("Uncoupled 3DCOM Trajectories")

plt.tight_layout()
plt.savefig('3dcom_visualization.png', dpi=300)
plt.show()
