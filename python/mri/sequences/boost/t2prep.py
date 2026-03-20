"""
T2 Preparation Module with MLEV4 Phase Cycling for BOOST Sequence

Implements T2prep using MLEV-4 scheme for bright-blood angiography.
The MLEV-4 scheme uses 4 RF pulses with specific phase cycling to achieve
T2-weighting while preserving steady-state magnetization.

Physics Background:
    T2prep attenuates signal based on T2 relaxation time.
    Blood has long T2 (~250 ms), while muscle has shorter T2 (~40 ms).
    This creates contrast between flowing blood and surrounding tissue.

    MLEV4 Phase Cycling:
    - Uses phase pattern: 0°, 90°, 180°, 270° for the four refocusing pulses
    - Provides robustness to B1 inhomogeneity
    - Creates symmetric echo formation

    At 0.55T:
    - Blood T2 ≈ 250 ms
    - Muscle T2 ≈ 40 ms
    - Fat T2 ≈ 100 ms

Parameters:
    - Duration: 50 ms total T2prep time
    - Flip angle: 155° for refocusing pulses
    - Pulse spacing: Evenly distributed over T2prep duration

Author: MRI Research Team
Date: 2026-03-20
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import pypulseq as pp


@dataclass
class T2PrepParameters:
    """Parameters for T2prep MLEV4."""
    duration: float = 50e-3  # Total T2prep duration (50 ms)
    flip_angle: float = 155  # Refocusing pulse flip angle in degrees
    pulse_duration: float = 2e-3  # Duration of each RF pulse
    time_btw_pulses: float = 12e-3  # Time between RF pulse centers
    use_mlev4: bool = True  # Use MLEV-4 phase cycling


class T2PrepMLEV4:
    """
    T2 Preparation using MLEV-4 phase cycling scheme.

    Creates T2-weighted contrast by applying 4 refocusing RF pulses.
    The MLEV-4 scheme uses specific phase cycling for robustness:
    Pulse 1: 0° phase
    Pulse 2: 90° phase
    Pulse 3: 180° phase
    Pulse 4: 270° phase

    The sequence structure is:
    [90° excitation] - [155° refocusing pulses with crushers] - [90° flip-back]

    Attributes:
        params: T2PrepParameters object
        block_events: List of sequence blocks

    Example:
        >>> params = T2PrepParameters(duration=50e-3, flip_angle=155)
        >>> t2prep = T2PrepMLEV4(params)
        >>> seq_blocks = t2prep.add_to_sequence(seq, system)
    """

    # MLEV-4 phase cycling pattern (in radians)
    MLEV4_PHASES = np.array([0, np.pi/2, np.pi, 3*np.pi/2])

    def __init__(self, params: Optional[T2PrepParameters] = None):
        """
        Initialize T2prep MLEV4 module.

        Args:
            params: T2PrepParameters object. Uses defaults if None.
        """
        self.params = params or T2PrepParameters()
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate T2prep parameters."""
        p = self.params

        if p.duration <= 0:
            raise ValueError(f"Duration must be positive, got {p.duration}")
        if p.flip_angle <= 0 or p.flip_angle >= 180:
            raise ValueError(f"Flip angle must be in (0, 180), got {p.flip_angle}")
        if p.pulse_duration <= 0:
            raise ValueError(f"Pulse duration must be positive, got {p.pulse_duration}")

        # Check if timing is feasible
        min_duration = 4 * p.pulse_duration
        if p.duration < min_duration:
            raise ValueError(f"Duration {p.duration*1e3:.1f} ms too short for "
                           f"4 pulses of {p.pulse_duration*1e3:.1f} ms each")

    def create_rf_pulse(self, flip_angle: float, phase: float,
                        system: pp.Opts) -> pp.RfSig:
        """
        Create RF pulse for T2prep.

        Args:
            flip_angle: Flip angle in degrees
            phase: Phase in radians
            system: pypulseq system limits

        Returns:
            pypulseq RF pulse object
        """
        # Create Sinc pulse with time-bandwidth product 4
        rf = pp.make_sinc_pulse(
            flip_angle=flip_angle * np.pi / 180,
            duration=self.params.pulse_duration,
            phase_offset=phase,
            system=system,
            time_bw_product=4,
        )
        return rf

    def create_crusher_gradient(self, area: float, duration: float,
                                system: pp.Opts) -> pp.Grad:
        """
        Create crusher gradient for signal dephasing.

        Crusher gradients before and after refocusing pulses prevent
        unwanted echoes from imperfect refocusing.

        Args:
            area: Gradient moment area (mT*s/m)
            duration: Crusher duration in seconds
            system: pypulseq system limits

        Returns:
            pypulseq gradient object
        """
        crusher = pp.make_trapezoid(
            channel='z',
            area=area,
            duration=duration,
            system=system
        )
        return crusher

    def add_to_sequence(self, seq: pp.Sequence,
                        system: pp.Opts) -> Tuple[pp.Sequence, float]:
        """
        Add T2prep blocks to sequence.

        Constructs the T2prep sequence:
        1. 90° excitation pulse
        2. 4× (155° refocusing pulse + crushers) with MLEV-4 phases
        3. 90° flip-back pulse

        Args:
            seq: pypulseq Sequence object to add blocks to
            system: pypulseq system limits

        Returns:
            Tuple of (modified sequence, duration in seconds)
        """
        p = self.params

        # Calculate timing
        n_pulses = 4  # MLEV-4 has 4 refocusing pulses
        time_between_pulses = p.duration / (n_pulses + 1)

        # Crusher parameters
        crusher_area = 10  # mT*ms/m - sufficient for dephasing
        crusher_duration = 2e-3  # 2 ms

        # Create initial 90° excitation pulse
        excitation = pp.make_sinc_pulse(
            flip_angle=np.pi/2,
            duration=p.pulse_duration,
            phase_offset=0,
            system=system,
            time_bw_product=4,
        )

        # Add excitation pulse
        seq.add_block(pp.make_rf_pulse(excitation.signal,
                                        flip_angle=90,
                                        phase_offset=0,
                                        system=system))

        # Add delay before first refocusing pulse
        delay1 = (time_between_pulses - p.pulse_duration) / 2
        seq.add_block(pp.make_delay(delay1))

        # Add 4 refocusing pulses with MLEV-4 phase cycling
        for i in range(4):
            # Get MLEV-4 phase
            phase = self.MLEV4_PHASES[i] if p.use_mlev4 else 0

            # Create refocusing pulse
            refocus = pp.make_sinc_pulse(
                flip_angle=p.flip_angle * np.pi / 180,
                duration=p.pulse_duration,
                phase_offset=phase,
                system=system,
                time_bw_product=4,
            )

            # Add crusher before pulse
            crusher_pre = self.create_crusher_gradient(crusher_area, crusher_duration, system)
            seq.add_block(crusher_pre)

            # Add refocusing pulse
            seq.add_block(pp.make_rf_pulse(refocus.signal,
                                           flip_angle=p.flip_angle,
                                           phase_offset=phase * 180 / np.pi,
                                           system=system))

            # Add crusher after pulse
            crusher_post = self.create_crusher_gradient(-crusher_area, crusher_duration, system)
            seq.add_block(crusher_post)

            # Add delay between pulses (except after last)
            if i < 3:
                delay_between = time_between_pulses - p.pulse_duration - 2 * crusher_duration
                seq.add_block(pp.make_delay(delay_between))

        # Add delay before flip-back
        delay2 = (time_between_pulses - p.pulse_duration) / 2
        seq.add_block(pp.make_delay(delay2))

        # Create 90° flip-back pulse (negative to return magnetization to z)
        flipback = pp.make_sinc_pulse(
            flip_angle=-np.pi/2,
            duration=p.pulse_duration,
            phase_offset=0,
            system=system,
            time_bw_product=4,
        )

        seq.add_block(pp.make_rf_pulse(flipback.signal,
                                        flip_angle=-90,
                                        phase_offset=0,
                                        system=system))

        return seq, p.duration

    def calculate_t2_weighting(self, t2: float, t1: float) -> float:
        """
        Calculate signal attenuation due to T2prep.

        For MLEV-4 T2prep, the signal is attenuated approximately as:
        S = exp(-TE_eff / T2) * (1 - exp(-TR_prep / T1))

        Args:
            t2: T2 relaxation time in seconds
            t1: T1 relaxation time in seconds

        Returns:
            Relative signal intensity (0 to 1)
        """
        # Effective echo time for T2prep
        te_eff = self.params.duration / 2

        # T2 decay
        t2_factor = np.exp(-te_eff / t2)

        # T1 recovery during preparation
        t1_factor = 1 - np.exp(-self.params.duration / t1)

        return t2_factor * t1_factor


def simulate_t2prep_signal(t2_values: dict, t1_values: dict,
                            params: Optional[T2PrepParameters] = None) -> dict:
    """
    Simulate signal evolution during T2prep for different tissues.

    Args:
        t2_values: Dictionary of tissue T2 values in seconds
                  e.g., {'blood': 250e-3, 'muscle': 40e-3, 'fat': 100e-3}
        t1_values: Dictionary of tissue T1 values in seconds
        params: T2PrepParameters object

    Returns:
        Dictionary of relative signals for each tissue
    """
    params = params or T2PrepParameters()
    t2prep = T2PrepMLEV4(params)

    signals = {}
    for tissue in t2_values:
        t2 = t2_values[tissue]
        t1 = t1_values.get(tissue, 1.0)  # Default T1 if not provided
        signals[tissue] = t2prep.calculate_t2_weighting(t2, t1)

    return signals


def create_boost_t2prep() -> T2PrepMLEV4:
    """
    Factory function to create T2prep for BOOST sequence.

    Configured specifically for carotid angiography at 0.55T:
    - 50 ms duration for adequate T2 contrast
    - 155° flip angle for robust refocusing
    - MLEV-4 phase cycling for B1 robustness

    Returns:
        T2PrepMLEV4 object configured for BOOST
    """
    params = T2PrepParameters(
        duration=50e-3,  # 50 ms
        flip_angle=155,  # 155 degrees
        pulse_duration=2e-3,  # 2 ms per pulse
        use_mlev4=True,
    )
    return T2PrepMLEV4(params)