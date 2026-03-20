"""
Bloch Simulation for BOOST Sequence Contrast Verification

Implements Bloch equation simulation to verify tissue contrast between
blood, fat, and muscle in the BOOST sequence. Simulates magnetization
evolution through preparation pulses and readout.

Physics Background:
    Bloch equations describe magnetization evolution:
    dM/dt = γ(M × B) - (Mx*M0/T2, My*M0/T2, (Mz-M0)/T1)

    For the BOOST sequence:
    1. T2prep: Attenuates signal based on T2
       - Blood (T2=250ms) retains ~90% signal
       - Muscle (T2=40ms) retains ~30% signal
    2. Inversion: Creates T1-weighted contrast
       - Blood recovers faster due to longer T1
    3. FatSat: Suppresses fat signal
    4. bSSFP readout: Steady-state signal depends on T1, T2

    Contrast mechanism:
    - Heartbeat 1: Bright blood (T2prep + inversion)
    - Heartbeat 2: Gray blood (no T2prep)

Author: MRI Research Team
Date: 2026-03-20
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


@dataclass
class TissueProperties:
    """Tissue relaxation properties at given field strength."""
    name: str
    T1: float  # seconds
    T2: float  # seconds
    M0: float = 1.0  # Relative signal

    def __post_init__(self):
        """Validate tissue properties."""
        if self.T1 <= 0:
            raise ValueError(f"T1 must be positive, got {self.T1}")
        if self.T2 <= 0:
            raise ValueError(f"T2 must be positive, got {self.T2}")


@dataclass
class SimulationParameters:
    """Parameters for Bloch simulation."""
    # Time steps
    dt: float = 0.1e-3  # seconds (0.1 ms)
    t2prep_duration: float = 50e-3  # seconds
    inversion_time: float = 70e-3  # seconds
    fatsat_duration: float = 14e-3  # seconds
    tr: float = 7.14e-3  # seconds
    num_shots: int = 24

    # Sequence
    field_strength: float = 0.55  # Tesla
    flip_angle: float = 90.0  # degrees

    # Default tissue properties at 0.55T
    blood: TissueProperties = None
    fat: TissueProperties = None
    muscle: TissueProperties = None

    def __post_init__(self):
        """Initialize default tissue properties."""
        if self.blood is None:
            self.blood = TissueProperties('blood', T1=1122e-3, T2=250e-3)
        if self.fat is None:
            self.fat = TissueProperties('fat', T1=183e-3, T2=100e-3)
        if self.muscle is None:
            self.muscle = TissueProperties('muscle', T1=750e-3, T2=40e-3)


class BlochSimulator:
    """
    Bloch equation simulator for BOOST sequence.

    Simulates magnetization evolution through:
    - T2prep MLEV4 preparation
    - Inversion pulse
    - FatSat
    - bSSFP readout

    Provides visualization of signal evolution and final contrast.

    Example:
        >>> params = SimulationParameters()
        >>> sim = BlochSimulator(params)
        >>> signals = sim.simulate_full_sequence()
        >>> sim.plot_signal_evolution()
    """

    # Gyromagnetic ratio (Hz/T)
    GAMMA = 42.577e6

    def __init__(self, params: Optional[SimulationParameters] = None):
        """
        Initialize Bloch simulator.

        Args:
            params: SimulationParameters object. Uses defaults if None.
        """
        self.params = params or SimulationParameters()
        self._initialize_state()

    def _initialize_state(self):
        """Initialize magnetization state for all tissues."""
        self.tissues = {
            'blood': self.params.blood,
            'fat': self.params.fat,
            'muscle': self.params.muscle,
        }

        # Magnetization state: [Mx, My, Mz] for each tissue
        self.states = {name: np.array([0.0, 0.0, 1.0]) for name in self.tissues}

        # Time vector
        self.time_points = []
        self.signal_evolution = {name: [] for name in self.tissues}

    def bloch_equations(self, t: float, M: np.ndarray,
                        T1: float, T2: float,
                        B: np.ndarray = np.array([0, 0, 0])) -> np.ndarray:
        """
        Bloch equations for magnetization evolution.

        dM/dt = γ(M × B) - (Mx/T2, My/T2, (Mz-M0)/T1)

        Args:
            t: Time (not used, for ODE solver)
            M: Magnetization vector [Mx, My, Mz]
            T1: T1 relaxation time
            T2: T2 relaxation time
            B: External field [Bx, By, Bz]

        Returns:
            Derivative dM/dt
        """
        Mx, My, Mz = M
        Bx, By, Bz = B

        # Precession term: γ(M × B)
        dMx_dt = self.GAMMA * (My * Bz - Mz * By) - Mx / T2
        dMy_dt = self.GAMMA * (Mz * Bx - Mx * Bz) - My / T2
        dMz_dt = self.GAMMA * (Mx * By - My * Bx) - (Mz - 1.0) / T1

        return np.array([dMx_dt, dMy_dt, dMz_dt])

    def apply_rf_pulse(self, M: np.ndarray,
                        flip_angle: float,
                        phase: float = 0.0) -> np.ndarray:
        """
        Apply RF pulse to magnetization.

        Rotates magnetization around the axis defined by phase angle.

        Args:
            M: Magnetization vector [Mx, My, Mz]
            flip_angle: Flip angle in degrees
            phase: Phase angle in degrees

        Returns:
            Rotated magnetization
        """
        alpha = np.radians(flip_angle)
        phi = np.radians(phase)

        # Rotation matrix around axis in xy-plane at angle phi
        # First rotate to align axis with x, then apply flip, then rotate back
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # Build rotation matrix
        # R = Rz(-phi) @ Rx(alpha) @ Rz(phi)
        R = np.array([
            [cos_alpha * cos_phi**2 + sin_phi**2,
             cos_phi * sin_phi * (cos_alpha - 1),
             sin_alpha * cos_phi],
            [cos_phi * sin_phi * (cos_alpha - 1),
             cos_alpha * sin_phi**2 + cos_phi**2,
             sin_alpha * sin_phi],
            [-sin_alpha * cos_phi,
             -sin_alpha * sin_phi,
             cos_alpha]
        ])

        return R @ M

    def apply_inversion(self, M: np.ndarray) -> np.ndarray:
        """
        Apply 180° inversion pulse.

        Args:
            M: Magnetization vector [Mx, My, Mz]

        Returns:
            Inverted magnetization
        """
        return self.apply_rf_pulse(M, flip_angle=180.0)

    def apply_excitation(self, M: np.ndarray,
                          flip_angle: float = 90.0) -> np.ndarray:
        """
        Apply excitation pulse.

        Args:
            M: Magnetization vector [Mx, My, Mz]
            flip_angle: Flip angle in degrees

        Returns:
            Magnetization after excitation
        """
        return self.apply_rf_pulse(M, flip_angle=flip_angle)

    def free_precession(self, M: np.ndarray,
                         T1: float, T2: float,
                         duration: float) -> np.ndarray:
        """
        Simulate free precession (T1/T2 relaxation).

        Args:
            M: Magnetization vector [Mx, My, Mz]
            T1: T1 relaxation time
            T2: T2 relaxation time
            duration: Duration of free precession

        Returns:
            Magnetization after free precession
        """
        Mx, My, Mz = M

        # T2 decay for transverse components
        Mx_new = Mx * np.exp(-duration / T2)
        My_new = My * np.exp(-duration / T2)

        # T1 recovery for longitudinal component
        Mz_new = 1.0 + (Mz - 1.0) * np.exp(-duration / T1)

        return np.array([Mx_new, My_new, Mz_new])

    def simulate_t2prep(self, M: np.ndarray,
                         T1: float, T2: float) -> Tuple[np.ndarray, float]:
        """
        Simulate T2prep MLEV4 sequence.

        Simplified model: applies T2 weighting to magnetization.

        Args:
            M: Initial magnetization [Mx, My, Mz]
            T1: T1 relaxation time
            T2: T2 relaxation time

        Returns:
            Tuple of (magnetization after T2prep, duration)
        """
        duration = self.params.t2prep_duration

        # T2prep effectively multiplies Mz by exp(-TE_eff/T2)
        # where TE_eff ≈ duration/2 for MLEV4
        te_eff = duration / 2

        # Apply T2 weighting
        M_new = M.copy()
        M_new[2] = M[2] * np.exp(-te_eff / T2)

        # Some T1 recovery during T2prep
        M_new[2] = M_new[2] + (1.0 - M_new[2]) * (1 - np.exp(-duration / T1)) * 0.5

        return M_new, duration

    def simulate_inversion(self, M: np.ndarray,
                            T1: float,
                            TI: float = None) -> Tuple[np.ndarray, float]:
        """
        Simulate inversion pulse and recovery.

        Args:
            M: Initial magnetization [Mx, My, Mz]
            T1: T1 relaxation time
            TI: Inversion time (uses params if None)

        Returns:
            Tuple of (magnetization after inversion, duration)
        """
        if TI is None:
            TI = self.params.inversion_time

        # Invert magnetization
        M_inv = self.apply_inversion(M)

        # Free precession during TI
        M_final = self.free_precession(M_inv, T1, T2=1.0, duration=TI)

        # Return only z-component after recovery (transverse dephased)
        return np.array([0.0, 0.0, M_final[2]]), TI

    def simulate_fatsat(self, M: np.ndarray,
                         tissue: TissueProperties) -> Tuple[np.ndarray, float]:
        """
        Simulate fat saturation.

        For fat: applies 180° pulse and spoils transverse
        For other tissues: minimal effect

        Args:
            M: Initial magnetization [Mx, My, Mz]
            tissue: Tissue properties

        Returns:
            Tuple of (magnetization after FatSat, duration)
        """
        duration = self.params.fatsat_duration

        # Fat suppression effect
        if tissue.name == 'fat':
            # Fat magnetization inverted and spoiled
            M_fat = self.apply_inversion(M)
            # Transverse spoiled, only Mz remains
            M_final = np.array([0.0, 0.0, M_fat[2]])
        else:
            # Other tissues minimally affected
            # Small T1 recovery during FatSat
            M_final = self.free_precession(M, tissue.T1, tissue.T2, duration)

        return M_final, duration

    def simulate_bssfp_shot(self, M: np.ndarray,
                             T1: float, T2: float,
                             flip_angle: float = None) -> Tuple[np.ndarray, float]:
        """
        Simulate single bSSFP TR.

        Simplified steady-state model for signal.

        Args:
            M: Initial magnetization [Mx, My, Mz]
            T1: T1 relaxation time
            T2: T2 relaxation time
            flip_angle: Flip angle in degrees

        Returns:
            Tuple of (magnetization after TR, signal magnitude, duration)
        """
        if flip_angle is None:
            flip_angle = self.params.flip_angle

        TR = self.params.tr
        alpha = np.radians(flip_angle)

        # Steady-state bSSFP signal (simplified)
        # S = M0 * sin(alpha/2) * sqrt(T2/T1) * (1 - E1) / (1 - E1*cos(alpha) - E2*(E1-1)*sin²(alpha)/(1+E2))
        E1 = np.exp(-TR / T1)
        E2 = np.exp(-TR / T2)

        # For on-resonance, simplified formula:
        # S = M0 * sin(alpha/2) / (1 + cos(alpha))  (approximately)
        # More accurate:
        numerator = np.sin(alpha) * (1 - E1)
        denominator = 1 - E1 * np.cos(alpha) + (E1 - E2) / (1 + E2)
        signal = M[2] * numerator / denominator

        # Update magnetization after RF
        M_after = self.apply_excitation(M, flip_angle)

        # Free precession during TR
        M_final = self.free_precession(M_after, T1, T2, TR)

        return M_final, abs(signal), TR

    def simulate_heartbeat1(self) -> Dict[str, List]:
        """
        Simulate Heartbeat 1 (Bright Blood).

        Sequence: T2prep → Inversion → FatSat → bSSFP readout

        Returns:
            Dictionary of signal evolution for each tissue
        """
        signals = {name: [] for name in self.tissues}
        times = []

        t = 0.0

        # Initialize to equilibrium
        states = {name: np.array([0.0, 0.0, 1.0]) for name in self.tissues}

        # T2prep
        for name, tissue in self.tissues.items():
            states[name], dur = self.simulate_t2prep(states[name], tissue.T1, tissue.T2)
        t += self.params.t2prep_duration
        times.append(t)
        for name in self.tissues:
            signals[name].append(abs(states[name][2]))

        # Inversion
        for name, tissue in self.tissues.items():
            states[name], dur = self.simulate_inversion(states[name], tissue.T1)
        t += self.params.inversion_time
        times.append(t)
        for name in self.tissues:
            signals[name].append(abs(states[name][2]))

        # FatSat
        for name, tissue in self.tissues.items():
            states[name], dur = self.simulate_fatsat(states[name], tissue)
        t += self.params.fatsat_duration
        times.append(t)
        for name in self.tissues:
            signals[name].append(abs(states[name][2]))

        # bSSFP readout
        for shot in range(self.params.num_shots):
            for name, tissue in self.tissues.items():
                states[name], sig, dur = self.simulate_bssfp_shot(
                    states[name], tissue.T1, tissue.T2
                )
                signals[name].append(sig)
            t += self.params.tr
            times.append(t)

        return {'times': times, 'signals': signals}

    def simulate_heartbeat2(self,
                             initial_states: Dict[str, np.ndarray] = None) -> Dict[str, List]:
        """
        Simulate Heartbeat 2 (Gray Blood).

        Sequence: FatSat → bSSFP readout

        Args:
            initial_states: Initial magnetization states (default: from heartbeat 1)

        Returns:
            Dictionary of signal evolution for each tissue
        """
        signals = {name: [] for name in self.tissues}
        times = []

        t = 0.0

        # Initialize states
        if initial_states is None:
            states = {name: np.array([0.0, 0.0, 1.0]) for name in self.tissues}
        else:
            states = initial_states.copy()

        # FatSat
        for name, tissue in self.tissues.items():
            states[name], dur = self.simulate_fatsat(states[name], tissue)
        t += self.params.fatsat_duration
        times.append(t)
        for name in self.tissues:
            signals[name].append(abs(states[name][2]))

        # bSSFP readout
        for shot in range(self.params.num_shots):
            for name, tissue in self.tissues.items():
                states[name], sig, dur = self.simulate_bssfp_shot(
                    states[name], tissue.T1, tissue.T2
                )
                signals[name].append(sig)
            t += self.params.tr
            times.append(t)

        return {'times': times, 'signals': signals}

    def simulate_full_sequence(self) -> Dict:
        """
        Simulate complete BOOST sequence (both heartbeats).

        Returns:
            Dictionary with times and signals for both heartbeats
        """
        # Heartbeat 1
        hb1 = self.simulate_heartbeat1()

        # Cardiac delay between heartbeats
        cardiac_delay = 900e-3  # Typical R-R interval
        hb1_duration = hb1['times'][-1]

        # Allow T1 recovery during cardiac delay
        states_after_hb1 = {name: np.array([0.0, 0.0, hb1['signals'][name][-1]])
                           for name in self.tissues}
        for name, tissue in self.tissues.items():
            states_after_hb1[name] = self.free_precession(
                states_after_hb1[name], tissue.T1, tissue.T2, cardiac_delay
            )

        # Heartbeat 2
        hb2 = self.simulate_heartbeat2(states_after_hb1)

        return {
            'heartbeat1': hb1,
            'heartbeat2': hb2,
            'cardiac_delay': cardiac_delay,
        }

    def calculate_contrast(self, simulation: Dict = None) -> Dict:
        """
        Calculate contrast between tissues.

        Args:
            simulation: Simulation results (runs if None)

        Returns:
            Dictionary with contrast metrics
        """
        if simulation is None:
            simulation = self.simulate_full_sequence()

        # Get final signals from each heartbeat
        hb1_signals = simulation['heartbeat1']['signals']
        hb2_signals = simulation['heartbeat2']['signals']

        # Mean signal during readout (last num_shots values)
        contrast = {}
        for tissue in self.tissues:
            hb1_mean = np.mean(hb1_signals[tissue][-self.params.num_shots:])
            hb2_mean = np.mean(hb2_signals[tissue][-self.params.num_shots:])

            contrast[tissue] = {
                'heartbeat1_signal': hb1_mean,
                'heartbeat2_signal': hb2_mean,
                'difference': hb1_mean - hb2_mean,
                'ratio': hb1_mean / hb2_mean if hb2_mean > 0 else np.inf,
            }

        # Blood-to-muscle contrast
        contrast['blood_muscle'] = {
            'heartbeat1': contrast['blood']['heartbeat1_signal'] / contrast['muscle']['heartbeat1_signal'],
            'heartbeat2': contrast['blood']['heartbeat2_signal'] / contrast['muscle']['heartbeat2_signal'],
        }

        # Blood-to-fat contrast
        contrast['blood_fat'] = {
            'heartbeat1': contrast['blood']['heartbeat1_signal'] / contrast['fat']['heartbeat1_signal'],
            'heartbeat2': contrast['blood']['heartbeat2_signal'] / contrast['fat']['heartbeat2_signal'],
        }

        return contrast

    def plot_signal_evolution(self, simulation: Dict = None,
                               save_path: str = None):
        """
        Plot signal evolution through BOOST sequence.

        Args:
            simulation: Simulation results (runs if None)
            save_path: Path to save figure (optional)
        """
        if simulation is None:
            simulation = self.simulate_full_sequence()

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Colors for tissues
        colors = {'blood': 'red', 'fat': 'green', 'muscle': 'blue'}

        # Heartbeat 1
        ax = axes[0]
        hb1 = simulation['heartbeat1']
        times_ms = np.array(hb1['times']) * 1e3  # Convert to ms

        for tissue, color in colors.items():
            signals = hb1['signals'][tissue]
            ax.plot(times_ms, signals, color=color, label=tissue.capitalize(), linewidth=2)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Signal (a.u.)')
        ax.set_title('Heartbeat 1: Bright Blood (T2prep + Inversion + FatSat + bSSFP)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add preparation markers
        ax.axvline(x=self.params.t2prep_duration*1e3, color='gray', linestyle='--', alpha=0.5, label='End T2prep')
        ax.axvline(x=(self.params.t2prep_duration + self.params.inversion_time)*1e3,
                   color='gray', linestyle=':', alpha=0.5, label='End Inversion')

        # Heartbeat 2
        ax = axes[1]
        hb2 = simulation['heartbeat2']
        times_ms = np.array(hb2['times']) * 1e3

        for tissue, color in colors.items():
            signals = hb2['signals'][tissue]
            ax.plot(times_ms, signals, color=color, label=tissue.capitalize(), linewidth=2)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Signal (a.u.)')
        ax.set_title('Heartbeat 2: Gray Blood (FatSat + bSSFP)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

        return fig

    def plot_contrast_bar(self, simulation: Dict = None,
                          save_path: str = None):
        """
        Plot bar chart of tissue contrast.

        Args:
            simulation: Simulation results (runs if None)
            save_path: Path to save figure (optional)
        """
        if simulation is None:
            simulation = self.simulate_full_sequence()

        contrast = self.calculate_contrast(simulation)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Tissue signals
        ax = axes[0]
        tissues = ['blood', 'fat', 'muscle']
        hb1_signals = [contrast[t]['heartbeat1_signal'] for t in tissues]
        hb2_signals = [contrast[t]['heartbeat2_signal'] for t in tissues]

        x = np.arange(len(tissues))
        width = 0.35

        ax.bar(x - width/2, hb1_signals, width, label='Heartbeat 1 (Bright)', color='darkred')
        ax.bar(x + width/2, hb2_signals, width, label='Heartbeat 2 (Gray)', color='lightcoral')

        ax.set_xlabel('Tissue')
        ax.set_ylabel('Signal Intensity (a.u.)')
        ax.set_title('BOOST Tissue Contrast')
        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in tissues])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Contrast ratios
        ax = axes[1]
        contrast_types = ['Blood/Muscle', 'Blood/Fat']
        hb1_contrast = [contrast['blood_muscle']['heartbeat1'],
                        contrast['blood_fat']['heartbeat1']]
        hb2_contrast = [contrast['blood_muscle']['heartbeat2'],
                        contrast['blood_fat']['heartbeat2']]

        x = np.arange(len(contrast_types))

        ax.bar(x - width/2, hb1_contrast, width, label='Heartbeat 1', color='darkblue')
        ax.bar(x + width/2, hb2_contrast, width, label='Heartbeat 2', color='lightblue')

        ax.set_xlabel('Contrast Type')
        ax.set_ylabel('Signal Ratio')
        ax.set_title('BOOST Contrast Ratios')
        ax.set_xticks(x)
        ax.set_xticklabels(contrast_types)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

        return fig


def run_bloch_simulation(field_strength: float = 0.55) -> Tuple[Dict, plt.Figure]:
    """
    Run complete Bloch simulation for BOOST sequence.

    Args:
        field_strength: Magnetic field strength in Tesla

    Returns:
        Tuple of (simulation results, figure)
    """
    params = SimulationParameters(field_strength=field_strength)
    sim = BlochSimulator(params)

    # Run simulation
    simulation = sim.simulate_full_sequence()

    # Calculate contrast
    contrast = sim.calculate_contrast(simulation)

    # Print results
    print("\n" + "="*60)
    print("BOOST Sequence Bloch Simulation Results")
    print("="*60)

    print("\nTissue Signals (Mean during readout):")
    for tissue in ['blood', 'fat', 'muscle']:
        print(f"  {tissue.capitalize()}:")
        print(f"    Heartbeat 1: {contrast[tissue]['heartbeat1_signal']:.3f}")
        print(f"    Heartbeat 2: {contrast[tissue]['heartbeat2_signal']:.3f}")
        print(f"    Ratio (HB1/HB2): {contrast[tissue]['ratio']:.2f}")

    print("\nContrast Ratios:")
    print(f"  Blood/Muscle:")
    print(f"    Heartbeat 1: {contrast['blood_muscle']['heartbeat1']:.2f}")
    print(f"    Heartbeat 2: {contrast['blood_muscle']['heartbeat2']:.2f}")
    print(f"  Blood/Fat:")
    print(f"    Heartbeat 1: {contrast['blood_fat']['heartbeat1']:.2f}")
    print(f"    Heartbeat 2: {contrast['blood_fat']['heartbeat2']:.2f}")

    print("="*60 + "\n")

    # Plot results
    fig1 = sim.plot_signal_evolution(simulation)
    fig2 = sim.plot_contrast_bar(simulation)

    return simulation, (fig1, fig2)