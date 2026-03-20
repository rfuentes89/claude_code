"""
Spiral Trajectory Module for BOOST Sequence

Implements variable-density spiral trajectories with golden-angle ordering
for undersampled bSSFP readout in carotid angiography.

Physics Background:
    - Spiral trajectories sample k-space in a rotating fashion
    - Golden-angle (111.25°) provides optimal incoherent sampling
    - Variable density improves image quality at edges
    - At 0.55T, field strength is lower, requiring careful gradient design

Key Parameters:
    - FOV: Field of view (200×208×88 mm³)
    - Resolution: ~0.96 mm³ isotropic
    - Undersampling: 5x acceleration
    - Off-resonance consideration: fat at ~224 Hz

Author: MRI Research Team
Date: 2026-03-20
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class SpiralParameters:
    """Parameters for spiral trajectory design."""
    fov: float = 200e-3  # Field of view in meters (200 mm)
    resolution: float = 0.96e-3  # Resolution in meters (0.96 mm)
    num_shots: int = 24  # Number of spiral shots
    samples_per_shot: int = 512  # ADC samples per shot
    undersampling_factor: float = 5.0  # Undersampling factor
    golden_angle_deg: float = 111.25  # Golden angle in degrees
    max_gradient: float = 40e-3  # Maximum gradient strength (T/m)
    max_slew: float = 150  # Maximum slew rate (T/m/s)
    spiral_type: str = 'variable_density'  # Type of spiral


class SpiralTrajectory:
    """
    Generates spiral trajectories with golden-angle ordering for BOOST sequence.

    The golden-angle approach ensures optimal incoherent k-space sampling
    across multiple shots, essential for compressed sensing reconstruction.

    Attributes:
        params: SpiralParameters object containing trajectory settings
        kx: k-space x coordinates (shots × samples)
        ky: k-space y coordinates (shots × samples)
        gx: Gradient waveform x (shots × samples)
        gy: Gradient waveform y (shots × samples)
        angles: Rotation angles for each shot

    Example:
        >>> params = SpiralParameters(fov=200e-3, resolution=0.96e-3)
        >>> spiral = SpiralTrajectory(params)
        >>> kx, ky, gx, gy = spiral.get_trajectory(shot_index=0)
    """

    # Golden ratio constant
    GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
    GOLDEN_ANGLE_RAD = np.pi * (3 - np.sqrt(5))  # ~111.25° in radians

    def __init__(self, params: Optional[SpiralParameters] = None):
        """
        Initialize spiral trajectory generator.

        Args:
            params: SpiralParameters object. Uses defaults if None.
        """
        self.params = params or SpiralParameters()
        self._validate_parameters()
        self._precompute_trajectory()

    def _validate_parameters(self):
        """Validate trajectory parameters."""
        p = self.params

        if p.fov <= 0:
            raise ValueError(f"FOV must be positive, got {p.fov}")
        if p.resolution <= 0 or p.resolution >= p.fov:
            raise ValueError(f"Resolution must be in (0, FOV), got {p.resolution}")
        if p.num_shots <= 0:
            raise ValueError(f"Number of shots must be positive, got {p.num_shots}")
        if p.undersampling_factor <= 0:
            raise ValueError(f"Undersampling factor must be positive, got {p.undersampling_factor}")

        # Warn if parameters seem unusual
        if p.max_gradient > 50e-3:
            warnings.warn(f"Gradient strength {p.max_gradient*1000:.1f} mT/m may exceed typical limits")
        if p.max_slew > 200:
            warnings.warn(f"Slew rate {p.max_slew:.0f} T/m/s may exceed typical limits")

    def _precompute_trajectory(self):
        """Precompute trajectory for all shots."""
        p = self.params

        # Calculate k-space extent
        self.kmax = 1 / (2 * p.resolution)  # Maximum k-space extent

        # Calculate number of full k-space samples (no undersampling)
        full_samples = int(np.ceil(p.fov / p.resolution))

        # Effective undersampling
        self.effective_shots = int(p.num_shots / p.undersampling_factor)

        # Generate base spiral (single shot)
        self._base_kx, self._base_ky = self._generate_base_spiral()

        # Generate gradient waveforms from k-space
        self._base_gx, self._base_gy = self._k_to_gradient(self._base_kx, self._base_ky)

        # Generate angles for all shots
        self.angles = self._generate_golden_angles(p.num_shots)

        # Precompute rotated trajectories
        self._kx = np.zeros((p.num_shots, p.samples_per_shot))
        self._ky = np.zeros((p.num_shots, p.samples_per_shot))
        self._gx = np.zeros((p.num_shots, p.samples_per_shot))
        self._gy = np.zeros((p.num_shots, p.samples_per_shot))

        for i in range(p.num_shots):
            self._kx[i], self._ky[i] = self._rotate_trajectory(
                self._base_kx, self._base_ky, self.angles[i]
            )
            self._gx[i], self._gy[i] = self._rotate_trajectory(
                self._base_gx, self._base_gy, self.angles[i]
            )

    def _generate_base_spiral(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate base spiral trajectory (unrotated).

        Uses Archimedean spiral with variable density.

        Returns:
            Tuple of (kx, ky) arrays for base spiral
        """
        p = self.params
        n = p.samples_per_shot

        # Time vector
        t = np.linspace(0, 1, n)

        # Archimedean spiral: r = a * theta
        # Variable density: use different rates for inner and outer regions

        # Maximum radius in k-space
        r_max = self.kmax

        # Variable density parameter (controls inner vs outer sampling)
        alpha = 1.5  # Higher = more samples at center

        # Radius progression with variable density
        r = r_max * np.power(t, 1/alpha)

        # Angular progression
        # Number of rotations depends on resolution and FOV
        n_rotations = self.kmax * p.fov  # Approximate number of rotations needed
        theta = 2 * np.pi * n_rotations * t

        # Convert to Cartesian coordinates
        kx = r * np.cos(theta)
        ky = r * np.sin(theta)

        return kx, ky

    def _k_to_gradient(self, kx: np.ndarray, ky: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert k-space trajectory to gradient waveform.

        The gradient is proportional to the time derivative of k-space:
        G = (gamma)^-1 * dk/dt

        For spiral trajectories, we need to ensure gradient limits are met.

        Args:
            kx: k-space x coordinates
            ky: k-space y coordinates

        Returns:
            Tuple of (gx, gy) gradient waveforms in T/m
        """
        p = self.params

        # Time step per sample (will be set by sequence)
        # Assume typical ADC dwell time
        dt = 4e-6  # 4 microseconds per sample

        # Gamma for protons (rad/s/T)
        gamma = 267.522e6  # rad/s/T

        # Calculate gradient from k-space derivative
        # k(t) = gamma * integral(G(t) dt) / (2*pi)
        # G(t) = dk/dt / (gamma / (2*pi))

        dkx = np.gradient(kx) / dt
        dky = np.gradient(ky) / dt

        # Gradient in T/m
        gx = dkx / (gamma / (2 * np.pi))
        gy = dky / (gamma / (2 * np.pi))

        # Clip to maximum gradient
        g_mag = np.sqrt(gx**2 + gy**2)
        scale = np.maximum(1, g_mag / p.max_gradient)
        gx = gx / scale
        gy = gy / scale

        return gx, gy

    def _generate_golden_angles(self, num_shots: int) -> np.ndarray:
        """
        Generate golden-angle rotation for each shot.

        Golden angle: ~111.25° provides optimal incoherent sampling
        for compressed sensing and parallel imaging.

        Args:
            num_shots: Number of spiral shots

        Returns:
            Array of rotation angles in radians
        """
        angles = self.GOLDEN_ANGLE_RAD * np.arange(num_shots)
        return angles % (2 * np.pi)

    def _rotate_trajectory(self, kx: np.ndarray, ky: np.ndarray,
                           angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate trajectory by given angle.

        Args:
            kx: k-space x coordinates
            ky: k-space y coordinates
            angle: Rotation angle in radians

        Returns:
            Tuple of rotated (kx, ky)
        """
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        kx_rot = kx * cos_a - ky * sin_a
        ky_rot = kx * sin_a + ky * cos_a
        return kx_rot, ky_rot

    def get_trajectory(self, shot_index: int) -> Tuple[np.ndarray, np.ndarray,
                                                         np.ndarray, np.ndarray]:
        """
        Get k-space and gradient trajectory for a specific shot.

        Args:
            shot_index: Index of spiral shot (0 to num_shots-1)

        Returns:
            Tuple of (kx, ky, gx, gy) for the specified shot

        Raises:
            IndexError: If shot_index is out of range
        """
        if shot_index < 0 or shot_index >= self.params.num_shots:
            raise IndexError(f"Shot index {shot_index} out of range [0, {self.params.num_shots-1}]")

        return (self._kx[shot_index], self._ky[shot_index],
                self._gx[shot_index], self._gy[shot_index])

    def get_all_trajectories(self) -> Tuple[np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray]:
        """
        Get all k-space and gradient trajectories.

        Returns:
            Tuple of (kx, ky, gx, gy) arrays (num_shots × samples_per_shot)
        """
        return self._kx, self._ky, self._gx, self._gy

    def get_adc_timing(self, shot_index: int, dwell_time: float = 4e-6) -> np.ndarray:
        """
        Get ADC timing for a specific shot.

        Args:
            shot_index: Index of spiral shot
            dwell_time: ADC dwell time in seconds

        Returns:
            Array of ADC sample times in seconds
        """
        return np.arange(self.params.samples_per_shot) * dwell_time

    def calculate_readout_duration(self, dwell_time: float = 4e-6) -> float:
        """
        Calculate total readout duration per shot.

        Args:
            dwell_time: ADC dwell time in seconds

        Returns:
            Readout duration in seconds
        """
        return self.params.samples_per_shot * dwell_time

    def visualize_trajectory(self, shots_to_plot: Optional[int] = None):
        """
        Visualize spiral trajectory in k-space.

        Args:
            shots_to_plot: Number of shots to plot. If None, plot all.
        """
        import matplotlib.pyplot as plt

        if shots_to_plot is None:
            shots_to_plot = min(self.params.num_shots, 24)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot k-space trajectory
        ax = axes[0]
        for i in range(shots_to_plot):
            kx, ky, _, _ = self.get_trajectory(i)
            ax.plot(kx * 1e-3, ky * 1e-3, alpha=0.5, linewidth=0.5)
        ax.set_xlabel('kx (cycles/mm)')
        ax.set_ylabel('ky (cycles/mm)')
        ax.set_title(f'Spiral k-space Trajectory\n({shots_to_plot} shots)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Plot gradient waveforms for first shot
        ax = axes[1]
        gx, gy = self._gx[0], self._gy[0]
        t = np.arange(len(gx)) * 4e-6 * 1e3  # Time in ms
        ax.plot(t, gx * 1e3, label='Gx (mT/m)')
        ax.plot(t, gy * 1e3, label='Gy (mT/m)')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Gradient (mT/m)')
        ax.set_title('Gradient Waveforms (First Shot)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig


def calculate_recommended_shots(fov: float, resolution: float,
                                undersampling: float = 5.0) -> int:
    """
    Calculate recommended number of spiral shots for given FOV and resolution.

    Args:
        fov: Field of view in meters
        resolution: Desired resolution in meters
        undersampling: Undersampling factor

    Returns:
        Recommended number of shots
    """
    # Nyquist sampling requirement
    nyquist_shots = int(np.ceil(fov / resolution))

    # Account for undersampling
    recommended = int(np.ceil(nyquist_shots / undersampling))

    return recommended


def create_boost_spiral(fov: float = 200e-3,
                        resolution: float = 0.96e-3,
                        num_shots: int = 24,
                        undersampling: float = 5.0) -> SpiralTrajectory:
    """
    Factory function to create BOOST-optimized spiral trajectory.

    Configures parameters specifically for carotid angiography at 0.55T.

    Args:
        fov: Field of view in meters (default 200 mm)
        resolution: Resolution in meters (default 0.96 mm)
        num_shots: Number of spiral shots (default 24)
        undersampling: Undersampling factor (default 5.0)

    Returns:
        SpiralTrajectory object configured for BOOST sequence
    """
    params = SpiralParameters(
        fov=fov,
        resolution=resolution,
        num_shots=num_shots,
        undersampling_factor=undersampling,
        max_gradient=40e-3,  # 40 mT/m typical for 0.55T
        max_slew=150,  # 150 T/m/s
    )

    return SpiralTrajectory(params)