"""
Inversion Pulse Module for BOOST Sequence

Implements adiabatic inversion pulse for blood signal nulling in BOOST sequence.
The inversion pulse flips magnetization 180° followed by an inversion time (TI)
to create optimal contrast.

Physics Background:
    Inversion recovery creates T1-weighted contrast:
    Mz(t) = M0 * (1 - 2*exp(-TI/T1))

    For blood at 0.55T (T1 ≈ 1122 ms):
    - TI = 70 ms provides partial recovery
    - Blood signal goes from -M0 toward +M0 during TI

    The inversion pulse in BOOST sequence:
    - Prepares blood for bright-blood contrast in Heartbeat 1
    - Blood magnetization is inverted before T2prep
    - After TI and T2prep, blood recovers to near-equilibrium

Parameters:
    - TI = 70 ms for 0.55T carotid imaging
    - Adiabatic pulse for uniform inversion over B1 variations
    - Crusher gradients after inversion for signal spoil

Author: MRI Research Team
Date: 2026-03-20
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import pypulseq as pp


@dataclass
class InversionParameters:
    """Parameters for inversion pulse."""
    ti: float = 70e-3  # Inversion time in seconds (70 ms)
    pulse_duration: float = 8e-3  # Duration of adiabatic inversion pulse
    crusher_area: float = 15.0  # Crusher gradient area (mT*ms/m)
    crusher_duration: float = 3e-3  # Crusher duration in seconds
    use_adiabatic: bool = True  # Use adiabatic pulse (recommended)


class InversionPulse:
    """
    Inversion pulse with crusher gradients for BOOST sequence.

    Creates a 180° inversion pulse followed by crusher gradients.
    The inversion time (TI) creates T1-weighted contrast.

    Attributes:
        params: InversionParameters object
        adiabatic_pulse: Whether using adiabatic pulse

    Example:
        >>> params = InversionParameters(ti=70e-3)
        >>> inv = InversionPulse(params)
        >>> seq, duration = inv.add_to_sequence(seq, system)
    """

    def __init__(self, params: Optional[InversionParameters] = None):
        """
        Initialize inversion pulse module.

        Args:
            params: InversionParameters object. Uses defaults if None.
        """
        self.params = params or InversionParameters()
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate inversion parameters."""
        p = self.params

        if p.ti <= 0:
            raise ValueError(f"Inversion time must be positive, got {p.ti}")
        if p.pulse_duration <= 0:
            raise ValueError(f"Pulse duration must be positive, got {p.pulse_duration}")
        if p.ti < p.pulse_duration:
            raise ValueError(f"TI ({p.ti*1e3:.1f} ms) must be >= pulse duration ({p.pulse_duration*1e3:.1f} ms)")

    def create_adiabatic_inversion(self, system: pp.Opts) -> pp.RfSig:
        """
        Create adiabatic inversion pulse (HS1).

        Adiabatic pulses provide uniform inversion over B1 variations,
        important at 0.55T where B1 homogeneity may be limited.

        Args:
            system: pypulseq system limits

        Returns:
            pypulseq RF pulse object
        """
        # Hyperbolic secant (HS1) adiabatic pulse
        # Time-bandwidth product determines bandwidth and selectivity
        duration = self.params.pulse_duration
        time_bw_product = 10  # Typical for adiabatic inversion

        # Create frequency-swept pulse
        rf = pp.make_adiabatic_pulse(
            pulse_type='hypsec',
            duration=duration,
            system=system,
            freq_offset=0,
            time_bw_product=time_bw_product,
        )

        return rf

    def create_simple_inversion(self, system: pp.Opts) -> pp.RfSig:
        """
        Create simple sinc inversion pulse.

        For cases where B1 uniformity is good, a simple sinc pulse
        can be used instead of adiabatic.

        Args:
            system: pypulseq system limits

        Returns:
            pypulseq RF pulse object
        """
        rf = pp.make_sinc_pulse(
            flip_angle=np.pi,  # 180 degrees
            duration=self.params.pulse_duration,
            phase_offset=0,
            system=system,
            time_bw_product=4,
        )

        return rf

    def create_crusher_gradient(self, system: pp.Opts) -> pp.Grad:
        """
        Create crusher gradient after inversion pulse.

        Crusher gradients dephase transverse magnetization
        after inversion to prevent unwanted signals.

        Args:
            system: pypulseq system limits

        Returns:
            pypulseq gradient object
        """
        crusher = pp.make_trapezoid(
            channel='z',
            area=self.params.crusher_area,
            duration=self.params.crusher_duration,
            system=system
        )
        return crusher

    def add_to_sequence(self, seq: pp.Sequence,
                        system: pp.Opts) -> Tuple[pp.Sequence, float]:
        """
        Add inversion pulse blocks to sequence.

        Constructs:
        1. Inversion pulse (adiabatic or sinc)
        2. Crusher gradients (both polarities for balance)
        3. Delay for inversion time

        Args:
            seq: pypulseq Sequence object
            system: pypulseq system limits

        Returns:
            Tuple of (modified sequence, total duration)
        """
        p = self.params

        # Create inversion pulse
        if p.use_adiabatic:
            rf = self.create_adiabatic_inversion(system)
        else:
            rf = self.create_simple_inversion(system)

        # Add inversion pulse
        seq.add_block(pp.make_rf_pulse(rf.signal if hasattr(rf, 'signal') else rf,
                                        flip_angle=180,
                                        phase_offset=0,
                                        system=system))

        # Add crusher gradients
        crusher_pos = self.create_crusher_gradient(system)
        crusher_neg = pp.make_trapezoid(
            channel='z',
            area=-p.crusher_area,
            duration=p.crusher_duration,
            system=system
        )

        seq.add_block(crusher_pos)
        seq.add_block(crusher_neg)

        # Add delay for inversion time
        delay_after_pulse = p.ti - p.pulse_duration - 2 * p.crusher_duration
        if delay_after_pulse > 0:
            seq.add_block(pp.make_delay(delay_after_pulse))

        total_duration = p.ti

        return seq, total_duration

    def calculate_inversion_recovery(self, t1: float, t: float) -> float:
        """
        Calculate magnetization after inversion recovery.

        Mz(t) = M0 * (1 - 2*exp(-t/T1))

        Args:
            t1: T1 relaxation time in seconds
            t: Time after inversion in seconds

        Returns:
            Relative z-magnetization (ranging from -1 to +1)
        """
        return 1 - 2 * np.exp(-t / t1)


def calculate_optimal_ti(t1_target: float,
                          target_recovery: float = 0.5) -> float:
    """
    Calculate optimal inversion time for desired signal recovery.

    Solves: Mz = 1 - 2*exp(-TI/T1) = target_recovery

    Args:
        t1_target: Target T1 value in seconds
        target_recovery: Desired recovery fraction (default 0.5)

    Returns:
        Optimal TI in seconds
    """
    # Solving: 1 - 2*exp(-TI/T1) = target
    # exp(-TI/T1) = (1 - target) / 2
    # TI = -T1 * ln((1 - target) / 2)

    ti = -t1_target * np.log((1 - target_recovery) / 2)
    return ti


def simulate_inversion_recovery(t1_values: dict,
                                 ti: float = 70e-3) -> dict:
    """
    Simulate inversion recovery for multiple tissues.

    Args:
        t1_values: Dictionary of tissue T1 values in seconds
        ti: Inversion time in seconds

    Returns:
        Dictionary of Mz/M0 values for each tissue
    """
    mz_values = {}
    for tissue, t1 in t1_values.items():
        mz_values[tissue] = 1 - 2 * np.exp(-ti / t1)

    return mz_values


def create_boost_inversion() -> InversionPulse:
    """
    Factory function to create inversion pulse for BOOST sequence.

    Configured for 0.55T carotid imaging:
    - TI = 70 ms for blood signal preparation
    - Adiabatic pulse for B1 robustness

    Returns:
        InversionPulse object configured for BOOST
    """
    params = InversionParameters(
        ti=70e-3,  # 70 ms
        pulse_duration=8e-3,  # 8 ms adiabatic pulse
        crusher_area=15.0,  # mT*ms/m
        crusher_duration=3e-3,  # 3 ms
        use_adiabatic=True,
    )
    return InversionPulse(params)