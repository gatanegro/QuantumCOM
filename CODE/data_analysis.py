"""
Comprehensive Data Analysis for UOFT-COM Paper
=============================================

This script performs detailed numerical analysis and extracts key results
on Quantum Entanglement and Bell's Inequality through UOFT and COM.

Author: Martin Doina
Date: 2025
"""

import numpy as np
import json
from oft_com_simulations import *

def comprehensive_analysis():
    """
    Perform comprehensive analysis and extract key numerical results.
    
    Returns:
        Dictionary containing all numerical results
    """
    print("Performing comprehensive data analysis...")
    
    results = {}
    
    # 1. Bell Inequality Analysis
    print("1. Analyzing Bell inequalities...")
    oft = OscillatoryFieldTheory(field_tension=0.1)
    angles = oft.optimal_bell_angles()
    
    results['bell_analysis'] = {
        'optimal_angles': angles,
        'oft_bell_parameter': oft.bell_parameter_oft(angles),
        'classical_bound': 2.0,
        'quantum_bound': 2 * np.sqrt(2),
        'violation_ratio': oft.bell_parameter_oft(angles) / 2.0
    }
    
    # Test different tension values
    tensions = np.linspace(0.01, 0.5, 50)
    bell_params = []
    for tension in tensions:
        oft_temp = OscillatoryFieldTheory(field_tension=tension)
        bell_param = oft_temp.bell_parameter_oft(angles)
        bell_params.append(bell_param)
    
    max_violation_idx = np.argmax(bell_params)
    results['bell_analysis']['max_violation'] = {
        'tension': tensions[max_violation_idx],
        'bell_parameter': bell_params[max_violation_idx],
        'violation_factor': bell_params[max_violation_idx] / 2.0
    }
    
    # 2. Collatz Sequence Analysis
    print("2. Analyzing Collatz sequences...")
    com = CollatzOctaveFramework(field_tension=0.05)
    
    # Analyze convergence for different starting values
    starting_values = [3, 5, 7, 8, 13, 17, 19, 23, 27]
    convergence_data = {}
    
    for start_val in starting_values:
        sequence = com.generate_field_collatz_sequence(start_val, max_steps=100)
        convergence_data[start_val] = {
            'sequence_length': len(sequence),
            'final_value': sequence[-1],
            'converged': abs(sequence[-1] - 1) < 0.5,
            'max_value': max(sequence),
            'octave_phases': [com.phase_mapping(int(abs(val))) for val in sequence[:10]]
        }
    
    results['collatz_analysis'] = {
        'convergence_data': convergence_data,
        'convergence_rate': sum(1 for data in convergence_data.values() if data['converged']) / len(convergence_data),
        'average_sequence_length': np.mean([data['sequence_length'] for data in convergence_data.values()])
    }
    
    # 3. FIELD Entanglement Analysis
    print("3. Analyzing FIELD entanglement...")
    
    # Multi-node entanglement
    field_tensor = FIELDTensor([5, 7, 8, 13], coupling=0.3, decoherence=0.1)
    field_tensor.evolve(100)
    
    # Compute pairwise Bell parameters
    sequences = []
    for i in range(4):
        seq = [state[i] for state in field_tensor.history]
        sequences.append(seq)
    
    pairwise_bells = {}
    for i in range(4):
        for j in range(i+1, 4):
            bell_param = compute_bell_parameter_collatz(sequences[i], sequences[j])
            pairwise_bells[f'{[5,7,8,13][i]}-{[5,7,8,13][j]}'] = bell_param
    
    # Compute correlation matrix
    correlation_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            correlation_matrix[i, j] = np.corrcoef(sequences[i], sequences[j])[0, 1]
    
    results['entanglement_analysis'] = {
        'pairwise_bell_parameters': pairwise_bells,
        'correlation_matrix': correlation_matrix.tolist(),
        'average_correlation': np.mean(correlation_matrix[np.triu_indices(4, k=1)]),
        'entanglement_measure': np.mean(list(pairwise_bells.values()))
    }
    
    # 4. Holographic Analysis
    print("4. Analyzing holographic correspondence...")
    
    holographic = HolographicProjection()
    
    # Generate structured bulk field
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    z = np.linspace(0.1, 3, 10)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    bulk_field = np.exp(-(X**2 + Y**2)/2) * np.cos(2*np.pi*Z)
    
    # Project to boundary
    hologram = holographic.project_to_boundary(bulk_field, z)
    correlator = holographic.compute_2pt_correlator(hologram)
    scaling_dim = holographic.extract_scaling_dimension(correlator)
    
    results['holographic_analysis'] = {
        'scaling_dimension': scaling_dim,
        'hologram_entropy': -np.sum(hologram * np.log(np.abs(hologram) + 1e-10)),
        'boundary_area': hologram.shape[0] * hologram.shape[1],
        'bulk_volume': bulk_field.shape[0] * bulk_field.shape[1] * bulk_field.shape[2]
    }
    
    # 5. CFT Analysis
    print("5. Analyzing conformal field theory...")
    
    cft = ConformalFieldTheory()
    ope_coeffs = cft.extract_ope_coefficients(hologram)
    theta_bins, angular_corr = cft.compute_angular_correlator(correlator)
    spins, power = cft.compute_spin_spectrum(angular_corr)
    
    dominant_spin = spins[np.argmax(power)]
    
    results['cft_analysis'] = {
        'central_charge': cft.c,
        'scaling_dimension': scaling_dim,
        'dominant_spin': int(dominant_spin),
        'ope_coefficients': ope_coeffs[:20].tolist(),
        'spin_spectrum': {
            'spins': spins.tolist(),
            'power': power.tolist()
        }
    }
    
    # 6. Decoherence Analysis
    print("6. Analyzing decoherence effects...")
    
    decoherence_strengths = np.linspace(0.01, 0.5, 20)
    decoherence_data = []
    
    for decoherence in decoherence_strengths:
        field_tensor_temp = FIELDTensor([5, 7], coupling=0.3, decoherence=decoherence)
        field_tensor_temp.evolve(50)
        
        seq1 = [state[0] for state in field_tensor_temp.history]
        seq2 = [state[1] for state in field_tensor_temp.history]
        
        bell_param = compute_bell_parameter_collatz(seq1, seq2)
        coherence = np.abs(np.corrcoef(seq1, seq2)[0, 1])
        
        decoherence_data.append({
            'decoherence_strength': decoherence,
            'bell_parameter': bell_param,
            'coherence': coherence
        })
    
    # Find critical decoherence
    bell_params = [data['bell_parameter'] for data in decoherence_data]
    critical_idx = np.where(np.array(bell_params) < 1.0)[0]
    critical_decoherence = decoherence_strengths[critical_idx[0]] if len(critical_idx) > 0 else None
    
    results['decoherence_analysis'] = {
        'decoherence_data': decoherence_data,
        'critical_decoherence': critical_decoherence,
        'coherence_decay_rate': -np.polyfit(decoherence_strengths, 
                                          [data['coherence'] for data in decoherence_data], 1)[0]
    }
    
    # 7. Phase Dynamics Analysis
    print("7. Analyzing phase dynamics...")
    
    oft_phase = OscillatoryFieldTheory(coupling_strength=0.3)
    n_oscillators = 8
    phases = np.random.uniform(0, 2*np.pi, n_oscillators)
    phase_history = [phases.copy()]
    
    for _ in range(200):
        phases = oft_phase.phase_dynamics(phases, dt=0.05, noise_strength=0.05)
        phase_history.append(phases.copy())
    
    phase_history = np.array(phase_history)
    
    # Compute order parameter
    order_param = []
    for t in range(len(phase_history)):
        complex_order = np.mean(np.exp(1j * phase_history[t]))
        order_param.append(np.abs(complex_order))
    
    # Synchronization time
    sync_threshold = 0.8
    sync_times = np.where(np.array(order_param) > sync_threshold)[0]
    sync_time = sync_times[0] if len(sync_times) > 0 else None
    
    results['phase_dynamics'] = {
        'final_order_parameter': order_param[-1],
        'synchronization_time': sync_time,
        'phase_coherence': np.mean(order_param[-50:]),  # Last 50 steps
        'coupling_strength': oft_phase.coupling
    }
    
    # 8. Statistical Summary
    print("8. Computing statistical summary...")
    
    results['statistical_summary'] = {
        'total_simulations': 1,
        'bell_violations_detected': int(results['bell_analysis']['oft_bell_parameter'] > 2.0),
        'collatz_convergence_rate': results['collatz_analysis']['convergence_rate'],
        'average_entanglement': results['entanglement_analysis']['entanglement_measure'],
        'holographic_scaling': results['holographic_analysis']['scaling_dimension'],
        'cft_central_charge': results['cft_analysis']['central_charge'],
        'critical_decoherence': results['decoherence_analysis']['critical_decoherence']
    }
    
    print("Data analysis complete!")
    return results

def save_results(results, filename='analysis_results.json'):
    """Save results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {filename}")

def print_key_results(results):
    """Print key numerical results."""
    print("\n" + "="*60)
    print("KEY NUMERICAL RESULTS")
    print("="*60)
    
    print(f"\n1. BELL INEQUALITY ANALYSIS:")
    print(f"   OFT Bell Parameter: {results['bell_analysis']['oft_bell_parameter']:.4f}")
    print(f"   Classical Bound: {results['bell_analysis']['classical_bound']:.4f}")
    print(f"   Violation Ratio: {results['bell_analysis']['violation_ratio']:.4f}")
    print(f"   Max Violation at T_FIELD = {results['bell_analysis']['max_violation']['tension']:.4f}")
    
    print(f"\n2. COLLATZ ANALYSIS:")
    print(f"   Convergence Rate: {results['collatz_analysis']['convergence_rate']:.2%}")
    print(f"   Average Sequence Length: {results['collatz_analysis']['average_sequence_length']:.1f}")
    
    print(f"\n3. FIELD ENTANGLEMENT:")
    print(f"   Average Correlation: {results['entanglement_analysis']['average_correlation']:.4f}")
    print(f"   Entanglement Measure: {results['entanglement_analysis']['entanglement_measure']:.4f}")
    
    print(f"\n4. HOLOGRAPHIC CORRESPONDENCE:")
    print(f"   Scaling Dimension Δ: {results['holographic_analysis']['scaling_dimension']:.4f}")
    print(f"   Boundary Entropy: {results['holographic_analysis']['hologram_entropy']:.2f}")
    
    print(f"\n5. CONFORMAL FIELD THEORY:")
    print(f"   Central Charge c: {results['cft_analysis']['central_charge']:.4f}")
    print(f"   Dominant Spin: {results['cft_analysis']['dominant_spin']}")
    
    print(f"\n6. DECOHERENCE:")
    print(f"   Critical Decoherence: {results['decoherence_analysis']['critical_decoherence']:.4f}")
    print(f"   Coherence Decay Rate: {results['decoherence_analysis']['coherence_decay_rate']:.4f}")
    
    print(f"\n7. PHASE DYNAMICS:")
    print(f"   Final Order Parameter: {results['phase_dynamics']['final_order_parameter']:.4f}")
    print(f"   Synchronization Time: {results['phase_dynamics']['synchronization_time']}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Run comprehensive analysis
    results = comprehensive_analysis()
    
    # Print key results
    print_key_results(results)
    
    # Save results
    save_results(results)

