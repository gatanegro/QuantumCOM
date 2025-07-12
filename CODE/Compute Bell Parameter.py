def compute_bell_param(seq1, seq2):
    min_len = min(len(seq1), len(seq2))
    phases1 = [(n % 9) * (2 * np.pi / 9) for n in seq1[:min_len]]
    phases2 = [(n % 9) * (2 * np.pi / 9) for n in seq2[:min_len]]
    return np.mean(np.cos(np.array(phases1) - np.array(phases2)))

C = compute_bell_param(seq7, seq8)
print(f"Bell Parameter C = {C:.3f}")  # Expected: C ≈ 1.2-1.3