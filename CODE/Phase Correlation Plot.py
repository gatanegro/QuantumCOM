phases7 = [(n % 9) * (2 * np.pi / 9) for n in seq7]
phases8 = [(n % 9) * (2 * np.pi / 9) for n in seq8]

plt.figure(figsize=(8, 6))
plt.scatter(phases7[:len(phases8)], phases8, c=range(len(phases8)), cmap='viridis', alpha=0.7)
plt.colorbar(label="Time Step")
plt.xlabel("Sequence n=7 Phase (radians)")
plt.ylabel("Sequence n=8 Phase (radians)")
plt.title(f"Collatz Phase Correlations (C = {C:.2f})")
plt.grid(True)
plt.savefig("collatz_bell.png", dpi=300, bbox_inches='tight')
plt.show()