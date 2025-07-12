"""
Oscillatory Field Theory (UOFT) and Collatz Octave Framework (COM) Simulations
=============================================================================

This module implements the mathematical framework for studying quantum entanglement
and Bell's inequality through the lens of UNIFIED Oscillatory Field Theory and the 
Collatz Octave Framework.

Author: Martin Doina
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.special as sp
from scipy.optimize import minimize
from scipy.fft import fft2, ifft2, fftshift
import pywt
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class OscillatoryFieldTheory:
    """
    Implementation of Oscillatory Field Theory (OFT) for quantum correlations.
    """
    
    def __init__(self, field_tension: float = 0.1, coupling_strength: float = 0.2):
        """
        Initialize OFT with field parameters.
        
        Args:
            field_tension: FIELD tension parameter T_FIELD
            coupling_strength: FIELD coupling constant
        """
        self.T_FIELD = field_tension
        self.coupling = coupling_strength
        self.c_field = 1.0  # FIELD propagation velocity
        self.omega_0 = 1.0  # Fundamental oscillation frequency
        
    def field_correlation(self, angle1: float, angle2: float) -> float:
        """
        Compute FIELD-mediated correlation function.
        
        Args:
            angle1, angle2: Measurement angles
            
        Returns:
            FIELD correlation E_FIELD(a,b)
        """
        return np.cos(angle1 - angle2) * np.exp(-self.T_FIELD * np.abs(angle1 - angle2))
    
    def bell_parameter_oft(self, angles: Dict[str, float]) -> float:
        """
        Compute Bell parameter in OFT framework.
        
        Args:
            angles: Dictionary with keys 'a', 'a_prime', 'b', 'b_prime'
            
        Returns:
            OFT Bell parameter S_OFT
        """
        a, a_p, b, b_p = angles['a'], angles['a_prime'], angles['b'], angles['b_prime']
        
        E_ab = self.field_correlation(a, b)
        E_apb = self.field_correlation(a_p, b)
        E_abp = self.field_correlation(a, b_p)
        E_apbp = self.field_correlation(a_p, b_p)
        
        return E_ab + E_apb + E_abp - E_apbp
    
    def optimal_bell_angles(self) -> Dict[str, float]:
        """
        Find optimal angles for maximum Bell violation in OFT.
        
        Returns:
            Dictionary of optimal angles
        """
        return {
            'a': np.pi/4,
            'a_prime': 3*np.pi/4,
            'b': np.pi/8,
            'b_prime': 5*np.pi/8
        }
    
    def phase_dynamics(self, phases: np.ndarray, dt: float = 0.01, 
                      noise_strength: float = 0.1) -> np.ndarray:
        """
        Evolve FIELD phase dynamics.
        
        Args:
            phases: Array of current phases
            dt: Time step
            noise_strength: Environmental noise strength
            
        Returns:
            Updated phases
        """
        N = len(phases)
        omega = np.ones(N)  # Natural frequencies
        
        # Kuramoto-like coupling
        coupling_term = np.zeros(N)
        for i in range(N):
            for j in range(N):
                if i != j:
                    coupling_term[i] += np.sin(phases[j] - phases[i])
        
        coupling_term *= self.coupling / N
        
        # Add noise
        noise = np.random.normal(0, noise_strength, N)
        
        # Update phases
        new_phases = phases + dt * (omega + coupling_term + noise)
        return new_phases % (2 * np.pi)


class CollatzOctaveFramework:
    """
    Implementation of Collatz Octave Framework (COM).
    """
    
    def __init__(self, field_tension: float = 0.05, noise_std: float = 0.1):
        """
        Initialize COM with parameters.
        
        Args:
            field_tension: FIELD tension for modified Collatz dynamics
            noise_std: Quantum fluctuation strength
        """
        self.T_FIELD = field_tension
        self.noise_std = noise_std
        self.layer_height = 1.0
        
    def octave_reduction(self, n: int) -> int:
        """
        Reduce number to octave (1-9).
        
        Args:
            n: Input number
            
        Returns:
            Octave-reduced value
        """
        return ((n - 1) % 9) + 1
    
    def phase_mapping(self, n: int) -> float:
        """
        Map number to phase angle.
        
        Args:
            n: Input number
            
        Returns:
            Phase angle in [0, 2π]
        """
        reduced = self.octave_reduction(n)
        return (2 * np.pi / 9) * reduced
    
    def field_collatz_step(self, n: float) -> float:
        """
        Single step of FIELD-coupled Collatz function.
        
        Args:
            n: Current value
            
        Returns:
            Next value with FIELD coupling
        """
        # Add quantum noise
        noise = np.random.normal(0, self.noise_std)
        
        if n % 2 == 0:
            return (n / 2) * np.exp(-self.T_FIELD * abs(n)) + noise
        else:
            return (3 * n + 1) * np.exp(-self.T_FIELD * abs(n)) + noise
    
    def generate_field_collatz_sequence(self, n: int, max_steps: int = 100) -> List[float]:
        """
        Generate FIELD-coupled Collatz sequence.
        
        Args:
            n: Starting number
            max_steps: Maximum iterations
            
        Returns:
            List of sequence values
        """
        sequence = [float(n)]
        current = float(n)
        
        for _ in range(max_steps):
            if abs(current - 1) < 0.1:  # Convergence criterion
                break
            current = self.field_collatz_step(current)
            sequence.append(current)
            
        return sequence
    
    def octave_spiral_coordinates(self, n: float, layer: int) -> Tuple[float, float, float]:
        """
        Map value to 3D octave spiral coordinates.
        
        Args:
            n: Value to map
            layer: Iteration layer
            
        Returns:
            (x, y, z) coordinates
        """
        phase = self.phase_mapping(int(abs(n)))
        radius = (layer + 1) * np.exp(-self.T_FIELD * layer)
        
        x = radius * np.cos(phase)
        y = radius * np.sin(phase)
        z = layer * self.layer_height
        
        return x, y, z


class FIELDTensor:
    """
    Multi-node FIELD tensor for entangled Collatz sequences.
    """
    
    def __init__(self, nodes: List[int], tension: float = 0.05, 
                 coupling: float = 0.3, decoherence: float = 0.1):
        """
        Initialize FIELD tensor.
        
        Args:
            nodes: List of starting Collatz values
            tension: FIELD tension parameter
            coupling: Inter-node coupling strength
            decoherence: Decoherence strength
        """
        self.nodes = [float(n) for n in nodes]
        self.tension = tension
        self.coupling = coupling
        self.decoherence = decoherence
        self.history = []
        self.com = CollatzOctaveFramework(tension)
        
    def step(self):
        """
        Single evolution step of FIELD tensor.
        """
        new_values = []
        
        for i, n in enumerate(self.nodes):
            # Base FIELD-Collatz step
            if n % 2 == 0:
                new_n = (n / 2) * np.exp(-self.tension * abs(n))
            else:
                new_n = (3 * n + 1) * np.exp(-self.tension * abs(n))
            
            # Inter-node interactions
            interaction = 0
            for j, other_n in enumerate(self.nodes):
                if i != j:
                    phase_diff = self.com.phase_mapping(int(abs(n))) - self.com.phase_mapping(int(abs(other_n)))
                    interaction += self.coupling * np.cos(phase_diff) * other_n
            
            # Decoherence
            phase_kick = np.random.normal(0, self.decoherence * abs(new_n))
            new_n *= np.exp(1j * phase_kick)
            new_n = np.real(new_n)  # Project back to real
            
            new_n += interaction
            new_values.append(new_n)
        
        self.nodes = new_values
        self.history.append(new_values.copy())
    
    def evolve(self, steps: int):
        """
        Evolve FIELD tensor for multiple steps.
        
        Args:
            steps: Number of evolution steps
        """
        for _ in range(steps):
            self.step()


class HolographicProjection:
    """
    Holographic projection between bulk COM and boundary OFT.
    """
    
    def __init__(self, ads_length: float = 1.0):
        """
        Initialize holographic projection.
        
        Args:
            ads_length: Characteristic AdS length scale
        """
        self.L_AdS = ads_length
        
    def holographic_kernel(self, x: float, y: float, z: float) -> float:
        """
        Holographic projection kernel.
        
        Args:
            x, y, z: Bulk coordinates
            
        Returns:
            Kernel value
        """
        if z <= 0:
            return 0
        return (1 / z**2) * np.exp(-z / self.L_AdS)
    
    def project_to_boundary(self, bulk_field: np.ndarray, z_coords: np.ndarray) -> np.ndarray:
        """
        Project bulk field to boundary.
        
        Args:
            bulk_field: 3D bulk field values
            z_coords: Z-coordinates for integration
            
        Returns:
            2D boundary field
        """
        boundary = np.zeros_like(bulk_field[:, :, 0])
        
        for i in range(bulk_field.shape[0]):
            for j in range(bulk_field.shape[1]):
                integral = 0
                for k in range(bulk_field.shape[2]):
                    if k < len(z_coords):
                        kernel = self.holographic_kernel(i, j, z_coords[k])
                        integral += kernel * bulk_field[i, j, k]
                boundary[i, j] = integral
                
        return boundary
    
    def compute_2pt_correlator(self, hologram: np.ndarray) -> np.ndarray:
        """
        Compute 2-point correlator from hologram.
        
        Args:
            hologram: 2D boundary field
            
        Returns:
            2-point correlation function
        """
        # Normalize
        O = hologram - np.mean(hologram)
        O /= np.std(O) if np.std(O) > 0 else 1
        
        # Compute autocorrelation via FFT
        fft_O = fft2(O)
        corr = np.real(ifft2(np.abs(fft_O)**2))
        corr = fftshift(corr)
        
        return corr
    
    def extract_scaling_dimension(self, correlator: np.ndarray) -> float:
        """
        Extract scaling dimension from correlator.
        
        Args:
            correlator: 2-point correlation function
            
        Returns:
            Scaling dimension Δ
        """
        center = np.array(correlator.shape) // 2
        y, x = np.indices(correlator.shape)
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(int)
        
        # Radial average
        radial_profile = np.bincount(r.ravel(), weights=correlator.ravel())
        counts = np.bincount(r.ravel())
        radial_profile = radial_profile / (counts + 1e-10)
        
        radii = np.arange(len(radial_profile))
        valid = (radii > 0) & (radial_profile > 0) & (radii < len(radii)//3)
        
        if np.sum(valid) < 3:
            return 1.0
        
        # Fit power law
        try:
            coeffs = np.polyfit(np.log(radii[valid]), np.log(radial_profile[valid]), 1)
            Delta = -coeffs[0] / 2
            return max(0.1, min(3.0, Delta))  # Reasonable bounds
        except:
            return 1.0


class ConformalFieldTheory:
    """
    Conformal Field Theory analysis tools.
    """
    
    def __init__(self, central_charge: float = None):
        """
        Initialize CFT analysis.
        
        Args:
            central_charge: Central charge (computed if None)
        """
        self.c = central_charge if central_charge else (3/2) * np.log(9)  # N_octave = 9
        
    def extract_ope_coefficients(self, hologram: np.ndarray, wavelet: str = 'db2', 
                                levels: int = 4) -> np.ndarray:
        """
        Extract OPE coefficients via wavelet decomposition.
        
        Args:
            hologram: 2D boundary field
            wavelet: Wavelet type
            levels: Decomposition levels
            
        Returns:
            Array of OPE coefficients
        """
        try:
            coeffs = pywt.wavedec2(hologram, wavelet, level=levels)
            
            # Flatten detail coefficients
            C = []
            for detail in coeffs[1:]:  # Skip approximation
                if isinstance(detail, tuple) and len(detail) == 3:
                    C.extend(detail[0].ravel())  # LH
                    C.extend(detail[1].ravel())  # HL
                    C.extend(detail[2].ravel())  # HH
            
            C = np.array(C)
            # Normalize
            if np.max(np.abs(C)) > 0:
                C = C / np.max(np.abs(C))
            
            return C
        except:
            return np.array([1.0])  # Fallback
    
    def compute_angular_correlator(self, correlator: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute angular dependence of correlator.
        
        Args:
            correlator: 2-point correlation function
            
        Returns:
            (theta_bins, angular_correlation)
        """
        center = np.array(correlator.shape) // 2
        y, x = np.indices(correlator.shape)
        theta = np.arctan2(y - center[1], x - center[0])
        
        theta_bins = np.linspace(-np.pi, np.pi, 36)
        angular_corr = np.zeros(len(theta_bins) - 1)
        
        for i in range(len(theta_bins) - 1):
            mask = (theta >= theta_bins[i]) & (theta < theta_bins[i + 1])
            if np.any(mask):
                angular_corr[i] = np.mean(correlator[mask])
        
        return theta_bins, angular_corr
    
    def compute_spin_spectrum(self, angular_corr: np.ndarray, n_modes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract spin spectrum from angular correlator.
        
        Args:
            angular_corr: Angular correlation function
            n_modes: Number of spin modes
            
        Returns:
            (spins, power_spectrum)
        """
        spins = np.arange(n_modes)
        signal = angular_corr - np.mean(angular_corr)
        
        # Pad signal to avoid issues
        if len(signal) < n_modes:
            signal = np.pad(signal, (0, n_modes - len(signal)), 'constant')
        
        fft_signal = np.fft.fft(signal)[:n_modes]
        power = np.abs(fft_signal)**2
        
        return spins, power


def compute_bell_parameter_collatz(seq1: List[float], seq2: List[float]) -> float:
    """
    Compute Collatz Bell parameter for two sequences.
    
    Args:
        seq1, seq2: Two Collatz sequences
        
    Returns:
        Collatz Bell parameter
    """
    max_len = max(len(seq1), len(seq2))
    s1 = np.array(seq1 + [0] * (max_len - len(seq1)))
    s2 = np.array(seq2 + [0] * (max_len - len(seq2)))
    
    # Normalize to unit vectors
    norm1 = np.linalg.norm(s1)
    norm2 = np.linalg.norm(s2)
    
    if norm1 > 0 and norm2 > 0:
        psi1 = s1 / norm1
        psi2 = s2 / norm2
        
        # Compute correlations at different time offsets
        C = 0
        for dt in [0, 1, 2]:
            shifted_psi2 = np.roll(psi2, dt)
            C += np.abs(np.dot(psi1, shifted_psi2))
        
        return C / 3  # Normalize
    else:
        return 0.0


def run_comprehensive_simulation():
    """
    Run comprehensive OFT-COM simulation.
    
    Returns:
        Dictionary containing all simulation results
    """
    print("Running comprehensive OFT-COM simulation...")
    
    # Initialize frameworks
    oft = OscillatoryFieldTheory(field_tension=0.1)
    com = CollatzOctaveFramework(field_tension=0.05)
    holographic = HolographicProjection()
    cft = ConformalFieldTheory()
    
    # 1. Bell correlations in OFT
    print("1. Computing Bell correlations...")
    angles = oft.optimal_bell_angles()
    bell_oft = oft.bell_parameter_oft(angles)
    
    # 2. FIELD-coupled Collatz sequences
    print("2. Generating FIELD-coupled Collatz sequences...")
    field_tensor = FIELDTensor([5, 7, 8, 13], coupling=0.2)
    field_tensor.evolve(50)
    
    # 3. Collatz Bell parameter
    print("3. Computing Collatz Bell parameter...")
    seq1 = [state[0] for state in field_tensor.history]
    seq2 = [state[1] for state in field_tensor.history]
    collatz_bell = compute_bell_parameter_collatz(seq1, seq2)
    
    # 4. Holographic projection
    print("4. Computing holographic projection...")
    # Create synthetic bulk field
    bulk_field = np.random.random((20, 20, 10))
    z_coords = np.linspace(0.1, 5, 10)
    hologram = holographic.project_to_boundary(bulk_field, z_coords)
    
    # 5. CFT analysis
    print("5. Performing CFT analysis...")
    correlator = holographic.compute_2pt_correlator(hologram)
    scaling_dim = holographic.extract_scaling_dimension(correlator)
    ope_coeffs = cft.extract_ope_coefficients(hologram)
    
    # 6. Angular analysis
    print("6. Computing angular correlations...")
    theta_bins, angular_corr = cft.compute_angular_correlator(correlator)
    spins, power = cft.compute_spin_spectrum(angular_corr)
    dominant_spin = spins[np.argmax(power)]
    
    results = {
        'bell_oft': bell_oft,
        'collatz_bell': collatz_bell,
        'scaling_dimension': scaling_dim,
        'dominant_spin': dominant_spin,
        'ope_coefficients': ope_coeffs[:10],  # First 10
        'field_tensor_history': field_tensor.history,
        'hologram': hologram,
        'correlator': correlator,
        'angular_correlation': (theta_bins, angular_corr),
        'spin_spectrum': (spins, power),
        'sequences': (seq1, seq2)
    }
    
    print("Simulation complete!")
    return results


if __name__ == "__main__":
    # Run the simulation
    results = run_comprehensive_simulation()
    
    # Print key results
    print(f"\nKey Results:")
    print(f"OFT Bell Parameter: {results['bell_oft']:.4f}")
    print(f"Collatz Bell Parameter: {results['collatz_bell']:.4f}")
    print(f"Scaling Dimension: {results['scaling_dimension']:.4f}")
    print(f"Dominant Spin: {results['dominant_spin']}")
    print(f"OPE Coefficients (first 5): {results['ope_coefficients'][:5]}")

