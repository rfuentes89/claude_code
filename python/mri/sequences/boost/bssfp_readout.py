"""
bSSFP Readout Module for BOOST Sequence

Implements balanced steady-state free precession (bSSFP) readout with
spiral trajectories for carotid angiography. bSSFP provides high signal
efficiency and good blood-to-tissue contrast.

Physics Background:
    bSSFP maintains steady-state magnetization with balanced gradients:
    - Gradient moments are zero at TR boundaries
    - Net gradient moment = 0 preserves steady state
    - Signal depends on T1, T2, and flip angle

    At 0.55T:
    - Blood: T1 ≈ 1122 ms, T2 ≈ 250 ms
    - Muscle: T1 ≈ 750 ms, T2 ≈ 40 ms
    - Fat: T1 ≈ 183 ms, T2 ≈ 100 ms

    Optimal flip angle for bSSFP:
    Ernst angle: cos(α_E) = exp(-TR/T1)
    For T1 ≈ 1000 ms, TR = 7.14 ms: α_E ≈ 90°

Parameters:
    - TR = 7.14 ms
    - TE = 3.57 ms (at TR/2 for optimal signal)
    - Flip angle = 90°
    - Bandwidth per pixel = 600 Hz
    - Spiral trajectory with golden-angle ordering

Author: MRI Research Team
Date: 2026-03-20
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import pypulseq as pp

from .spiral_trajectory import SpiralTrajectory, SpiralParameters, create_boost_spiral


@dataclass
class bSSFPParameters:
    """Parameters for bSSFP readout."""
    tr: float = 7.14e-3  # Repetition time in seconds (7.14 ms)
    te: float = 3.57e-3  # Echo time in seconds (3.57 ms)
    flip_angle: float = 90.0  # Flip angle in degrees
    bandwidth: float = 600.0  # Bandwidth per pixel in Hz
    num_shots: int = 24  # Number of spiral shots
    samples_per_shot: int = 512  # ADC samples per shot
    dwell_time: float = 4e-3  # ADC dwell time in ms (4 us)
    rf_duration: float = 1e-3  # RF pulse duration in seconds
    use_spiral: bool = True  # Use spiral readout


class bSSFPReadout:
    """
    bSSFP readout with spiral trajectory for BOOST sequence.

    Creates balanced SSFP sequence blocks with spiral gradient waveforms.
    Maintains gradient moment balance for steady-state preservation.

    Attributes:
        params: bSSFPParameters object
        spiral: SpiralTrajectory object for readout

    Example:
        >>> params = bSSFPParameters(tr=7.14e-3, flip_angle=90)
        >>> readout = bSSFPReadout(params)
        >>> seq, duration = readout.add_to_sequence(seq, system, shot_index=0)
    """

    def __init__(self, params: Optional[bSSFPParameters] = None,
                 spiral: Optional[SpiralTrajectory] = None):
        """
        Initialize bSSFP readout module.

        Args:
            params: bSSFPParameters object. Uses defaults if None.
            spiral: SpiralTrajectory object. Created from params if None.
        """
        self.params = params or bSSFPParameters()
        self._validate_parameters()

        # Create spiral trajectory if not provided
        if spiral is None:
            spiral_params = SpiralParameters(
                fov=200e-3,  # 200 mm
                resolution=0.96e-3,  # 0.96 mm
                num_shots=self.params.num_shots,
                samples_per_shot=self.params.samples_per_shot,
                undersampling_factor=5.0,
            )
            self.spiral = SpiralTrajectory(spiral_params)
        else:
            self.spiral = spiral

    def _validate_parameters(self):
        """Validate bSSFP parameters."""
        p = self.params

        if p.tr <= 0:
            raise ValueError(f"TR must be positive, got {p.tr}")
        if p.te < 0 or p.te > p.tr:
            raise ValueError(f"TE must be in [0, TR], got TE={p.te}, TR={p.tr}")
        if p.flip_angle <= 0 or p.flip_angle > 180:
            raise ValueError(f"Flip angle must be in (0, 180], got {p.flip_angle}")
        if p.bandwidth <= 0:
            raise ValueError(f"Bandwidth must be positive, got {p.bandwidth}")

    def calculate_ernst_angle(self, t1: float) -> float:
        """
        Calculate Ernst angle for given T1.

        cos(α_E) = exp(-TR/T1)

        Args:
            t1: T1 relaxation time in seconds

        Returns:
            Ernst angle in degrees
        """
        if t1 <= 0:
            raise ValueError(f"T1 must be positive, got {t1}")

        cos_alpha = np.exp(-self.params.tr / t1)
        alpha = np.arccos(cos_alpha) * 180 / np.pi
        return alpha

    def create_rf_pulse(self, system: pp.Opts) -> pp.RfSig:
        """
        Create RF excitation pulse for bSSFP.

        Uses Sinc pulse with appropriate flip angle.

        Args:
            system: pypulseq system limits

        Returns:
            pypulseq RF pulse object
        """
        p = self.params

        rf = pp.make_sinc_pulse(
            flip_angle=p.flip_angle * np.pi / 180,
            duration=p.rf_duration,
            phase_offset=0,
            system=system,
            time_bw_product=4,
        )

        return rf

    def create_spiral_readout(self, shot_index: int,
                               system: pp.Opts) -> Tuple[pp.Grad, pp.Grad, pp.Adc]:
        """
        Create spiral readout gradients and ADC.

        Args:
            shot_index: Index of spiral shot (0 to num_shots-1)
            system: pypulseq system limits

        Returns:
            Tuple of (gx, gy, adc) for the specified shot
        """
        # Get spiral trajectory for this shot
        kx, ky, gx, gy = self.spiral.get_trajectory(shot_index)

        # Calculate ADC timing
        dwell_time = self.params.dwell_time * 1e-3  # Convert ms to seconds
        num_samples = self.params.samples_per_shot

        # Create gradient waveforms
        # Note: pypulseq uses arbitrary gradient waveforms
        gx_grad = pp.make_arbitrary_grad(
            channel='x',
            wave=gx,
            system=system,
            delay=0,
        )

        gy_grad = pp.make_arbitrary_grad(
            channel='y',
            wave=gy,
            system=system,
            delay=0,
        )

        # Create ADC
        adc = pp.make_adc(
            num_samples=num_samples,
            duration=num_samples * dwell_time,
            delay=0,
            system=system,
        )

        return gx_grad, gy_grad, adc

    def create_balance_gradients(self, system: pp.Opts) -> Tuple[pp.Grad, pp.Grad]:
        """
        Create balance gradients to maintain steady state.

        Balance gradients ensure net gradient moment = 0 over TR.
        This is essential for bSSFP steady-state preservation.

        Args:
            system: pypulseq system limits

        Returns:
            Tuple of (gx_balance, gy_balance) gradient objects
        """
        p = self.params

        # For spiral trajectories, we need to rewind the gradient moment
        # The balance gradient area = -integral of readout gradient

        # Get total gradient moment from spiral
        kx, ky, gx, gy = self.spiral.get_trajectory(0)

        # Calculate moments (integral of gradient over time)
        dt = p.dwell_time * 1e-3  # seconds
        moment_x = np.trapz(gx, dx=dt)
        moment_y = np.trapz(gy, dx=dt)

        # Create balance gradients with opposite moment
        gx_balance = pp.make_trapezoid(
            channel='x',
            area=-moment_x,
            system=system,
        )

        gy_balance = pp.make_trapezoid(
            channel='y',
            area=-moment_y,
            system=system,
        )

        return gx_balance, gy_balance

    def add_to_sequence(self, seq: pp.Sequence,
                        system: pp.Opts,
                        shot_index: int) -> Tuple[pp.Sequence, float]:
        """
        Add bSSFP readout block to sequence.

        Constructs a single TR:
        1. RF excitation
        2. Spiral readout with ADC
        3. Balance gradients (if needed)
        4. Delay to complete TR

        Args:
            seq: pypulseq Sequence object
            system: pypulseq system limits
            shot_index: Index of spiral shot

        Returns:
            Tuple of (modified sequence, duration in seconds)
        """
        p = self.params

        # Calculate timing
        dwell_time = p.dwell_time * 1e-3  # Convert to seconds
        readout_duration = p.samples_per_shot * dwell_time

        # RF excitation
        rf = self.create_rf_pulse(system)
        seq.add_block(pp.make_rf_pulse(
            rf.signal if hasattr(rf, 'signal') else rf,
            flip_angle=p.flip_angle,
            phase_offset=0,
            system=system
        ))

        # Delay to TE
        te_delay = p.te - p.rf_duration / 2
        if te_delay > 0:
            seq.add_block(pp.make_delay(te_delay))

        # Spiral readout
        gx, gy, adc = self.create_spiral_readout(shot_index, system)

        # Add readout block (combined Gx, Gy, ADC)
        seq.add_block(gx, gy, adc)

        # Balance gradients
        # For bSSFP, we need zero net gradient moment over TR
        # This is achieved by gradient rewinding
        gx_balance, gy_balance = self.create_balance_gradients(system)
        seq.add_block(gx_balance, gy_balance)

        # Delay to complete TR
        current_duration = p.rf_duration + readout_duration + te_delay
        tr_remaining = p.tr - current_duration
        if tr_remaining > 0:
            seq.add_block(pp.make_delay(tr_remaining))

        return seq, p.tr

    def add_shots(self, seq: pp.Sequence,
                   system: pp.Opts,
                   num_shots: Optional[int] = None,
                   start_shot: int = 0) -> Tuple[pp.Sequence, float]:
        """
        Add multiple bSSFP readout shots to sequence.

        Args:
            seq: pypulseq Sequence object
            system: pypulseq system limits
            num_shots: Number of shots to add (default: all shots)
            start_shot: Starting shot index

        Returns:
            Tuple of (modified sequence, total duration in seconds)
        """
        if num_shots is None:
            num_shots = self.params.num_shots

        total_duration = 0
        for i in range(num_shots):
            shot_index = (start_shot + i) % self.params.num_shots
            seq, duration = self.add_to_sequence(seq, system, shot_index)
            total_duration += duration

        return seq, total_duration

    def calculate_signal(self, t1: float, t2: float,
                         flip_angle: Optional[float] = None) -> float:
        """
        Calculate bSSFP steady-state signal for given tissue.

        The steady-state signal for bSSFP is:
        S = M0 * sin(α) * (1 - E1) / (1 - E1*cos(α) - (E2-E1)*sin²(α)/(1+E2))

        where E1 = exp(-TR/T1), E2 = exp(-TR/T2)

        Simplified for on-resonance:
        S ≈ M0 * sin(α/2) * sqrt(T2/T1)

        Args:
            t1: T1 relaxation time in seconds
            t2: T2 relaxation time in seconds
            flip_angle: Flip angle in degrees (uses params if None)

        Returns:
            Relative steady-state signal intensity
        """
        p = self.params
        alpha = (flip_angle or p.flip_angle) * np.pi / 180

        E1 = np.exp(-p.tr / t1)
        E2 = np.exp(-p.tr / t2)

        # Steady-state signal (simplified on-resonance)
        numerator = np.sin(alpha) * (1 - E1)
        denominator = 1 - E1 * np.cos(alpha) + (E1 - E2) / (1 + E2)

        signal = numerator / denominator

        return signal

    def get_trajectory(self, shot_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get k-space trajectory for a specific shot.

        Args:
            shot_index: Index of spiral shot

        Returns:
            Tuple of (kx, ky) k-space coordinates
        """
        kx, ky, _, _ = self.spiral.get_trajectory(shot_index)
        return kx, ky


def create_boost_bssfp_readout() -> bSSFPReadout:
    """
    Factory function to create bSSFP readout for BOOST sequence.

    Configured for 0.55T carotid imaging:
    - TR = 7.14 ms
    - TE = 3.57 ms (TR/2)
    - Flip angle = 90°
    - 24 spiral shots with 5x undersampling

    Returns:
        bSSFPReadout object configured for BOOST
    """
    params = bSSFPParameters(
        tr=7.14e-3,  # 7.14 ms
        te=3.57e-3,  # 3.57 ms (TR/2)
        flip_angle=90.0,  # 90 degrees
        bandwidth=600.0,  # 600 Hz/pixel
        num_shots=24,
        samples_per_shot=512,
        dwell_time=4e-3,  # 4 us
        rf_duration=1e-3,  # 1 ms
        use_spiral=True,
    )

    spiral = create_boost_spiral()
    return bSSFPReadout(params, spiral)


def simulate_bssfp_contrast(t1_values: dict, t2_values: dict,
                             flip_angle: float = 90.0,
                             tr: float = 7.14e-3) -> dict:
    """
    Simulate bSSFP contrast for different tissues.

    Args:
        t1_values: Dictionary of tissue T1 values in seconds
        t2_values: Dictionary of tissue T2 values in seconds
        flip_angle: Flip angle in degrees
        tr: Repetition time in seconds

    Returns:
        Dictionary of relative signals for each tissue
    """
    params = bSSFPParameters(tr=tr, flip_angle=flip_angle)
    readout = bSSFPReadout(params)

    signals = {}
    for tissue in t1_values:
        t1 = t1_values[tissue]
        t2 = t2_values.get(tissue, t1 * 0.1)  # Approximate T2 if not provided
        signals[tissue] = readout.calculate_signal(t1, t2)

    return signals