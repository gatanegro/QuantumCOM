import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bars

def field_collatz(n, steps=100, tension=0.05, noise=0.01):
    sequence = []
    for _ in range(steps):
        if n == 1:
            break
        if n % 2 == 0:
            n = int(n // 2 * np.exp(-tension * n)) + np.random.normal(0, noise)
        else:
            n = int((3 * n + 1) * np.exp(-tension * n)) + np.random.normal(0, noise)
        sequence.append(n)
    return sequence

# Generate sequences
np.random.seed(42)  # For reproducibility
seq7 = field_collatz(7, steps=100)
seq8 = field_collatz(8, steps=100)