"""
Visualization Generation for UOFT-COM Paper
==========================================

This script generates all the plots and visualizations 
on Quantum Entanglement and Bell's Inequality through UOFT and COM.

Author: Martin Doina
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from oft_com_simulations import *
import os

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_output_directory():
    """Create output directory for plots."""
    if not os.path.exists('plots'):
        os.makedirs('plots')

def plot_bell_correlations():
    """Generate Bell correlation plots."""
    print("Generating Bell correlation plots...")
    
    oft = OscillatoryFieldTheory(field_tension=0.1)
    
    # Plot 1: FIELD correlation vs angle difference
    angle_diffs = np.linspace(0, np.pi, 100)
    correlations = [oft.field_correlation(0, diff) for diff in angle_diffs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(angle_diffs, correlations, 'b-', linewidth=2, label='FIELD Correlation')
    plt.plot(angle_diffs, np.cos(angle_diffs), 'r--', linewidth=2, label='Quantum Mechanical')
    plt.xlabel('Angle Difference |a - b| (radians)')
    plt.ylabel('Correlation E(a,b)')
    plt.title('FIELD-Mediated vs Quantum Mechanical Correlations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/bell_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Bell parameter vs FIELD tension
    tensions = np.linspace(0.01, 0.5, 50)
    bell_params = []
    
    for tension in tensions:
        oft_temp = OscillatoryFieldTheory(field_tension=tension)
        angles = oft_temp.optimal_bell_angles()
        bell_param = oft_temp.bell_parameter_oft(angles)
        bell_params.append(bell_param)
    
    plt.figure(figsize=(10, 6))
    plt.plot(tensions, bell_params, 'g-', linewidth=2)
    plt.axhline(y=2, color='r', linestyle='--', linewidth=2, label='Classical Bound')
    plt.axhline(y=2*np.sqrt(2), color='b', linestyle='--', linewidth=2, label='Quantum Bound')
    plt.xlabel('FIELD Tension Parameter')
    plt.ylabel('Bell Parameter S')
    plt.title('Bell Parameter vs FIELD Tension')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/bell_vs_tension.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_collatz_octave_3d():
    """Generate 3D Collatz octave visualizations."""
    print("Generating 3D Collatz octave plots...")
    
    com = CollatzOctaveFramework(field_tension=0.05)
    
    # Generate sequences for multiple starting values
    starting_values = [5, 7, 8, 13, 17]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, start_val in enumerate(starting_values):
        sequence = com.generate_field_collatz_sequence(start_val, max_steps=30)
        
        # Map to 3D coordinates
        x_coords, y_coords, z_coords = [], [], []
        for layer, value in enumerate(sequence):
            x, y, z = com.octave_spiral_coordinates(value, layer)
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
        
        # Plot trajectory
        ax.plot(x_coords, y_coords, z_coords, color=colors[i], 
                linewidth=2, alpha=0.8, label=f'n={start_val}')
        ax.scatter(x_coords, y_coords, z_coords, color=colors[i], s=30, alpha=0.6)
    
    ax.set_xlabel('X (Re[Ψ])')
    ax.set_ylabel('Y (Im[Ψ])')
    ax.set_zlabel('Layer (Iteration)')
    ax.set_title('3D Collatz Octave Spiral with FIELD Decay')
    ax.legend()
    plt.tight_layout()
    plt.savefig('plots/collatz_3d_spiral.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_field_entanglement():
    """Generate FIELD entanglement visualizations."""
    print("Generating FIELD entanglement plots...")
    
    # Multi-node FIELD tensor evolution
    field_tensor = FIELDTensor([5, 7, 8, 13], coupling=0.3, decoherence=0.1)
    field_tensor.evolve(100)
    
    # Plot 1: Time evolution of node values
    plt.figure(figsize=(12, 8))
    
    for i in range(4):
        values = [state[i] for state in field_tensor.history]
        plt.subplot(2, 2, i+1)
        plt.plot(values, linewidth=2)
        plt.title(f'Node {[5,7,8,13][i]} Evolution')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/field_entanglement_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Cross-correlations between nodes
    correlations = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            seq_i = [state[i] for state in field_tensor.history]
            seq_j = [state[j] for state in field_tensor.history]
            correlations[i, j] = np.corrcoef(seq_i, seq_j)[0, 1]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0,
                xticklabels=[5,7,8,13], yticklabels=[5,7,8,13])
    plt.title('FIELD-Mediated Cross-Correlations')
    plt.xlabel('Node')
    plt.ylabel('Node')
    plt.tight_layout()
    plt.savefig('plots/field_correlations_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_holographic_analysis():
    """Generate holographic analysis plots."""
    print("Generating holographic analysis plots...")
    
    # Create holographic projection
    holographic = HolographicProjection()
    
    # Generate synthetic bulk field with structure
    x = np.linspace(-2, 2, 30)
    y = np.linspace(-2, 2, 30)
    z = np.linspace(0.1, 3, 15)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Structured bulk field (spiral pattern)
    bulk_field = np.exp(-(X**2 + Y**2)/2) * np.cos(2*np.pi*Z) * np.sin(np.arctan2(Y, X) * 3)
    
    # Project to boundary
    hologram = holographic.project_to_boundary(bulk_field, z)
    
    # Compute 2-point correlator
    correlator = holographic.compute_2pt_correlator(hologram)
    
    # Plot hologram and correlator
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # Hologram
    im1 = ax1.imshow(hologram, cmap='viridis', extent=[-2, 2, -2, 2])
    ax1.set_title('Holographic Boundary Field')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1)
    
    # 2-point correlator
    im2 = ax2.imshow(correlator, cmap='RdBu', extent=[-1, 1, -1, 1])
    ax2.set_title('2-Point Correlator ⟨O(x)O(y)⟩')
    ax2.set_xlabel('Δx')
    ax2.set_ylabel('Δy')
    plt.colorbar(im2, ax=ax2)
    
    # Radial profile
    center = np.array(correlator.shape) // 2
    y_idx, x_idx = np.indices(correlator.shape)
    r = np.sqrt((x_idx - center[0])**2 + (y_idx - center[1])**2)
    r = r.astype(int)
    
    radial_profile = np.bincount(r.ravel(), weights=correlator.ravel())
    counts = np.bincount(r.ravel())
    radial_profile = radial_profile / (counts + 1e-10)
    radii = np.arange(len(radial_profile))
    
    valid = (radii > 0) & (radial_profile > 0) & (radii < len(radii)//2)
    ax3.loglog(radii[valid], radial_profile[valid], 'bo-', markersize=4)
    ax3.set_xlabel('Distance r')
    ax3.set_ylabel('⟨O(r)O(0)⟩')
    ax3.set_title('Radial Correlation Profile')
    ax3.grid(True, alpha=0.3)
    
    # Extract scaling dimension
    scaling_dim = holographic.extract_scaling_dimension(correlator)
    ax3.text(0.05, 0.95, f'Δ = {scaling_dim:.3f}', transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Angular correlator
    cft = ConformalFieldTheory()
    theta_bins, angular_corr = cft.compute_angular_correlator(correlator)
    ax4.plot(theta_bins[:-1], angular_corr, 'ro-', markersize=4)
    ax4.set_xlabel('Angle θ (radians)')
    ax4.set_ylabel('⟨O(θ)O(0)⟩')
    ax4.set_title('Angular Correlation')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/holographic_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_cft_analysis():
    """Generate CFT analysis plots."""
    print("Generating CFT analysis plots...")
    
    # Generate hologram for CFT analysis
    holographic = HolographicProjection()
    cft = ConformalFieldTheory()
    
    # Create structured hologram
    x = np.linspace(-1, 1, 50)
    y = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(x, y)
    hologram = np.exp(-(X**2 + Y**2)) * np.cos(3*np.arctan2(Y, X)) + 0.1*np.random.random((50, 50))
    
    # CFT analysis
    correlator = holographic.compute_2pt_correlator(hologram)
    ope_coeffs = cft.extract_ope_coefficients(hologram)
    theta_bins, angular_corr = cft.compute_angular_correlator(correlator)
    spins, power = cft.compute_spin_spectrum(angular_corr)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # OPE coefficients
    ax1.bar(range(len(ope_coeffs[:20])), ope_coeffs[:20], alpha=0.7)
    ax1.set_xlabel('Operator Index')
    ax1.set_ylabel('OPE Coefficient')
    ax1.set_title('Operator Product Expansion Coefficients')
    ax1.grid(True, alpha=0.3)
    
    # Spin spectrum
    ax2.bar(spins, power, alpha=0.7, color='orange')
    dominant_spin = spins[np.argmax(power)]
    ax2.axvline(dominant_spin, color='red', linestyle='--', linewidth=2,
                label=f'Dominant: s={dominant_spin}')
    ax2.set_xlabel('Spin s')
    ax2.set_ylabel('Power')
    ax2.set_title('Spin Spectrum (Fourier Modes)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Angular correlator
    ax3.plot(theta_bins[:-1], angular_corr, 'go-', markersize=4)
    ax3.set_xlabel('Angle θ (radians)')
    ax3.set_ylabel('⟨O(θ)O(0)⟩')
    ax3.set_title('Angular Dependence of Correlator')
    ax3.grid(True, alpha=0.3)
    
    # Wavelet decomposition visualization
    coeffs = pywt.wavedec2(hologram, 'db2', level=3)
    if len(coeffs) > 1 and isinstance(coeffs[1], tuple):
        LH = coeffs[1][0]  # Horizontal features
        im4 = ax4.imshow(LH, cmap='RdBu')
        ax4.set_title('Wavelet LH (Horizontal Features)')
        plt.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('plots/cft_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_decoherence_analysis():
    """Generate decoherence analysis plots."""
    print("Generating decoherence analysis plots...")
    
    # Study decoherence effects
    decoherence_strengths = np.linspace(0.01, 0.5, 20)
    bell_parameters = []
    coherence_measures = []
    
    for decoherence in decoherence_strengths:
        # FIELD tensor with varying decoherence
        field_tensor = FIELDTensor([5, 7], coupling=0.3, decoherence=decoherence)
        field_tensor.evolve(50)
        
        # Compute Bell parameter
        seq1 = [state[0] for state in field_tensor.history]
        seq2 = [state[1] for state in field_tensor.history]
        bell_param = compute_bell_parameter_collatz(seq1, seq2)
        bell_parameters.append(bell_param)
        
        # Compute coherence measure
        coherence = np.abs(np.corrcoef(seq1, seq2)[0, 1])
        coherence_measures.append(coherence)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bell parameter vs decoherence
    ax1.plot(decoherence_strengths, bell_parameters, 'bo-', linewidth=2, markersize=6)
    ax1.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Classical Threshold')
    ax1.set_xlabel('Decoherence Strength')
    ax1.set_ylabel('Collatz Bell Parameter')
    ax1.set_title('Bell Violation vs Decoherence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Coherence vs decoherence
    ax2.plot(decoherence_strengths, coherence_measures, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Decoherence Strength')
    ax2.set_ylabel('Sequence Coherence')
    ax2.set_title('FIELD Coherence Decay')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/decoherence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_phase_space_analysis():
    """Generate phase space analysis plots."""
    print("Generating phase space analysis plots...")
    
    # Phase dynamics simulation
    oft = OscillatoryFieldTheory(coupling_strength=0.3)
    
    # Initialize random phases
    n_oscillators = 8
    phases = np.random.uniform(0, 2*np.pi, n_oscillators)
    phase_history = [phases.copy()]
    
    # Evolve phases
    for _ in range(200):
        phases = oft.phase_dynamics(phases, dt=0.05, noise_strength=0.05)
        phase_history.append(phases.copy())
    
    phase_history = np.array(phase_history)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # Phase evolution
    for i in range(n_oscillators):
        ax1.plot(phase_history[:, i], alpha=0.7, label=f'Oscillator {i+1}')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Phase (radians)')
    ax1.set_title('FIELD Phase Evolution')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Phase space (first two oscillators)
    ax2.plot(phase_history[:, 0], phase_history[:, 1], 'b-', alpha=0.7)
    ax2.scatter(phase_history[0, 0], phase_history[0, 1], color='green', s=100, label='Start')
    ax2.scatter(phase_history[-1, 0], phase_history[-1, 1], color='red', s=100, label='End')
    ax2.set_xlabel('Phase 1 (radians)')
    ax2.set_ylabel('Phase 2 (radians)')
    ax2.set_title('Phase Space Trajectory')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Order parameter
    order_param = []
    for t in range(len(phase_history)):
        complex_order = np.mean(np.exp(1j * phase_history[t]))
        order_param.append(np.abs(complex_order))
    
    ax3.plot(order_param, 'g-', linewidth=2)
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Order Parameter |⟨e^{iθ}⟩|')
    ax3.set_title('FIELD Synchronization')
    ax3.grid(True, alpha=0.3)
    
    # Phase difference distribution
    phase_diffs = []
    for t in range(len(phase_history)):
        for i in range(n_oscillators):
            for j in range(i+1, n_oscillators):
                diff = np.abs(phase_history[t, i] - phase_history[t, j])
                diff = min(diff, 2*np.pi - diff)  # Wrap to [0, π]
                phase_diffs.append(diff)
    
    ax4.hist(phase_diffs, bins=30, alpha=0.7, density=True, color='purple')
    ax4.set_xlabel('Phase Difference (radians)')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('Phase Difference Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/phase_space_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_all_plots():
    """Generate all plots for the paper."""
    print("Generating all visualizations for OFT-COM paper...")
    
    create_output_directory()
    
    plot_bell_correlations()
    plot_collatz_octave_3d()
    plot_field_entanglement()
    plot_holographic_analysis()
    plot_cft_analysis()
    plot_decoherence_analysis()
    plot_phase_space_analysis()
    
    print("All visualizations generated successfully!")
    print("Plots saved in 'plots/' directory:")
    for filename in os.listdir('plots'):
        print(f"  - {filename}")

if __name__ == "__main__":
    generate_all_plots()

