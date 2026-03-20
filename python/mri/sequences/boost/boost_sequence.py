"""
BOOST Sequence Assembly for Carotid Angiography at 0.55T

Assembles the complete BOOST sequence (Bright-blood Optimized Angiography
using Single-shot Time-varied preparation) from the modular components.

Sequence Structure:
    Heartbeat 1 (Bright Blood):
        1. T2prep MLEV4 (50 ms, flip angle 155°)
        2. Inversion pulse (TI = 70 ms)
        3. FatSat spectral (180° pulse + crushers)
        4. bSSFP readout (24 spiral shots, golden-angle)

    Heartbeat 2 (Gray Blood):
        1. FatSat spectral (180° pulse + crushers)
        2. bSSFP readout (24 spiral shots, golden-angle)

Physics Background:
    The BOOST sequence exploits:
    1. T2prep: Blood has long T2 (~250 ms), attenuates short-T2 tissues
    2. Inversion: Creates contrast between fresh and stationary blood
    3. FatSat: Suppresses fat signal for better blood visualization
    4. bSSFP: High signal efficiency, good blood-to-tissue contrast

    At 0.55T:
    - Lower field allows different contrast mechanisms
    - Reduced SAR enables longer preparation pulses
    - Longer T1 times require adjusted timing

Parameters:
    - FOV: 200×208×88 mm³
    - Resolution: 0.96 mm³ isotropic
    - TR: 7.14 ms
    - TE: 3.57 ms
    - Flip angle: 90°
    - Undersampling: 5x

Author: MRI Research Team
Date: 2026-03-20
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass, field
import pypulseq as pp

from .spiral_trajectory import SpiralTrajectory, SpiralParameters, create_boost_spiral
from .t2prep import T2PrepMLEV4, T2PrepParameters, create_boost_t2prep
from .inversion_pulse import InversionPulse, InversionParameters, create_boost_inversion
from .fat_sat import FatSatSpectral, FatSatParameters, create_boost_fatsat
from .bssfp_readout import bSSFPReadout, bSSFPParameters, create_boost_bssfp_readout


@dataclass
class BOOSTParameters:
    """Complete parameters for BOOST sequence."""
    # Geometry
    fov: Tuple[float, float, float] = (200e-3, 208e-3, 88e-3)  # meters
    resolution: float = 0.96e-3  # meters

    # Timing
    tr: float = 7.14e-3  # seconds
    te: float = 3.57e-3  # seconds

    # bSSFP
    flip_angle: float = 90.0  # degrees
    num_shots: int = 24
    samples_per_shot: int = 512

    # T2prep
    t2prep_duration: float = 50e-3  # seconds
    t2prep_flip_angle: float = 155.0  # degrees

    # Inversion
    inversion_time: float = 70e-3  # seconds

    # FatSat
    fat_frequency_offset: float = 224.0  # Hz at 0.55T

    # Field
    field_strength: float = 0.55  # Tesla

    # Tissue properties (at 0.55T)
    t1_blood: float = 1122e-3  # seconds
    t1_fat: float = 183e-3  # seconds
    t1_muscle: float = 750e-3  # seconds
    t2_blood: float = 250e-3  # seconds
    t2_fat: float = 100e-3  # seconds
    t2_muscle: float = 40e-3  # seconds


class BOOSTSequence:
    """
    Complete BOOST sequence for carotid angiography at 0.55T.

    Assembles all preparation modules and readout into a complete sequence.
    Exports to Pulseq format (.seq) for execution on scanners.

    Attributes:
        params: BOOSTParameters object
        seq: pypulseq Sequence object
        spiral: SpiralTrajectory for readout
        t2prep: T2PrepMLEV4 module
        inversion: InversionPulse module
        fatsat: FatSatSpectral module
        bssfp: bSSFPReadout module

    Example:
        >>> boost = BOOSTSequence()
        >>> seq = boost.build_sequence()
        >>> seq.write('boost_carotid.seq')
    """

    def __init__(self, params: Optional[BOOSTParameters] = None):
        """
        Initialize BOOST sequence.

        Args:
            params: BOOSTParameters object. Uses defaults if None.
        """
        self.params = params or BOOSTParameters()
        self._validate_parameters()
        self._initialize_modules()

        # Create pypulseq sequence
        self.seq = pp.Sequence()

        # System limits for 0.55T scanner
        self.system = pp.Opts(
            max_grad=40,  # 40 mT/m
            max_slew=150,  # 150 T/m/s
            rf_dead_time=20e-6,  # 20 us
            adc_dead_time=10e-6,  # 10 us
            grad_raster_time=10e-6,  # 10 us
        )

    def _validate_parameters(self):
        """Validate BOOST sequence parameters."""
        p = self.params

        # Check geometry
        for dim, fov_dim in zip(['x', 'y', 'z'], p.fov):
            if fov_dim <= 0:
                raise ValueError(f"FOV {dim} must be positive, got {fov_dim}")
        if p.resolution <= 0:
            raise ValueError(f"Resolution must be positive, got {p.resolution}")

        # Check timing
        if p.tr <= 0:
            raise ValueError(f"TR must be positive, got {p.tr}")
        if p.te < 0 or p.te > p.tr:
            raise ValueError(f"TE must be in [0, TR], got TE={p.te}, TR={p.tr}")

        # Check tissue properties
        for tissue, (t1, t2) in [
            ('blood', (p.t1_blood, p.t2_blood)),
            ('fat', (p.t1_fat, p.t2_fat)),
            ('muscle', (p.t1_muscle, p.t2_muscle))
        ]:
            if t1 <= 0:
                raise ValueError(f"T1_{tissue} must be positive, got {t1}")
            if t2 <= 0:
                raise ValueError(f"T2_{tissue} must be positive, got {t2}")

    def _initialize_modules(self):
        """Initialize all sequence modules."""
        p = self.params

        # Spiral trajectory
        self.spiral = create_boost_spiral(
            fov=p.fov[0],  # Use x-dimension for now
            resolution=p.resolution,
            num_shots=p.num_shots,
            undersampling=5.0,
        )

        # T2prep
        self.t2prep = create_boost_t2prep()

        # Inversion
        self.inversion = create_boost_inversion()

        # FatSat
        self.fatsat = create_boost_fatsat(p.field_strength)

        # bSSFP readout
        self.bssfp = create_boost_bssfp_readout()

    def add_heartbeat1(self) -> float:
        """
        Add Heartbeat 1 blocks to sequence.

        Structure:
        1. T2prep MLEV4
        2. Inversion pulse
        3. FatSat
        4. bSSFP readout (24 shots)

        Returns:
            Total duration in seconds
        """
        duration = 0.0

        # T2prep
        self.seq, t2prep_dur = self.t2prep.add_to_sequence(self.seq, self.system)
        duration += t2prep_dur

        # Inversion
        self.seq, inv_dur = self.inversion.add_to_sequence(self.seq, self.system)
        duration += inv_dur

        # FatSat
        self.seq, fatsat_dur = self.fatsat.add_to_sequence(self.seq, self.system)
        duration += fatsat_dur

        # bSSFP readout
        self.seq, readout_dur = self.bssfp.add_shots(
            self.seq, self.system, num_shots=self.params.num_shots
        )
        duration += readout_dur

        return duration

    def add_heartbeat2(self) -> float:
        """
        Add Heartbeat 2 blocks to sequence.

        Structure:
        1. FatSat
        2. bSSFP readout (24 shots)

        Returns:
            Total duration in seconds
        """
        duration = 0.0

        # FatSat
        self.seq, fatsat_dur = self.fatsat.add_to_sequence(self.seq, self.system)
        duration += fatsat_dur

        # bSSFP readout (continuing from heartbeat 1)
        start_shot = self.params.num_shots  # Continue shot numbering
        self.seq, readout_dur = self.bssfp.add_shots(
            self.seq, self.system,
            num_shots=self.params.num_shots,
            start_shot=start_shot
        )
        duration += readout_dur

        return duration

    def build_sequence(self) -> pp.Sequence:
        """
        Build complete BOOST sequence.

        Constructs both heartbeats and returns the complete sequence.

        Returns:
            pypulseq Sequence object
        """
        # Reset sequence
        self.seq = pp.Sequence()

        # Add heartbeat 1 (bright blood)
        hb1_duration = self.add_heartbeat1()
        print(f"Heartbeat 1 duration: {hb1_duration*1e3:.1f} ms")

        # Add delay between heartbeats (simulate cardiac cycle)
        # Typical R-R interval: 800-1000 ms
        rr_interval = 900e-3  # 900 ms
        hb1_minus_prep = hb1_duration - 50e-3 - 70e-3  # Subtract T2prep and inversion time
        cardiac_delay = rr_interval - hb1_minus_prep
        if cardiac_delay > 0:
            self.seq.add_block(pp.make_delay(cardiac_delay))

        # Add heartbeat 2 (gray blood)
        hb2_duration = self.add_heartbeat2()
        print(f"Heartbeat 2 duration: {hb2_duration*1e3:.1f} ms")

        return self.seq

    def export(self, filename: str):
        """
        Export sequence to Pulseq .seq file.

        Args:
            filename: Output filename (.seq extension recommended)
        """
        self.seq.write(filename)

    def calculate_total_duration(self) -> float:
        """
        Calculate total sequence duration.

        Returns:
            Total duration in seconds
        """
        p = self.params

        # Heartbeat 1
        t2prep_dur = p.t2prep_duration
        inv_dur = p.inversion_time
        fatsat_dur = 14e-3  # Approximate FatSat duration
        readout_dur = p.num_shots * p.tr

        hb1_duration = t2prep_dur + inv_dur + fatsat_dur + readout_dur

        # Heartbeat 2
        hb2_duration = fatsat_dur + readout_dur

        # Cardiac delay
        rr_interval = 900e-3  # Typical R-R interval

        total = hb1_duration + rr_interval + hb2_duration

        return total

    def get_sequence_info(self) -> Dict:
        """
        Get detailed information about the sequence.

        Returns:
            Dictionary with sequence parameters and timing
        """
        p = self.params

        info = {
            'sequence': 'BOOST',
            'field_strength': f"{p.field_strength} T",
            'geometry': {
                'fov': f"{p.fov[0]*1e3:.1f} x {p.fov[1]*1e3:.1f} x {p.fov[2]*1e3:.1f} mm",
                'resolution': f"{p.resolution*1e3:.2f} mm",
            },
            'timing': {
                'tr': f"{p.tr*1e3:.2f} ms",
                'te': f"{p.te*1e3:.2f} ms",
                'flip_angle': f"{p.flip_angle}°",
            },
            'readout': {
                'num_shots': p.num_shots,
                'samples_per_shot': p.samples_per_shot,
                'undersampling': '5x',
                'trajectory': 'spiral golden-angle',
            },
            'preparation': {
                't2prep': f"{p.t2prep_duration*1e3:.0f} ms MLEV4",
                'inversion_time': f"{p.inversion_time*1e3:.0f} ms",
                'fatsat': f"spectral {p.fat_frequency_offset} Hz offset",
            },
            'tissue_properties': {
                'blood': f"T1={p.t1_blood*1e3:.0f} ms, T2={p.t2_blood*1e3:.0f} ms",
                'fat': f"T1={p.t1_fat*1e3:.0f} ms, T2={p.t2_fat*1e3:.0f} ms",
                'muscle': f"T1={p.t1_muscle*1e3:.0f} ms, T2={p.t2_muscle*1e3:.0f} ms",
            },
        }

        return info

    def print_sequence_info(self):
        """Print sequence information to console."""
        info = self.get_sequence_info()

        print("\n" + "="*60)
        print(f"BOOST Sequence for Carotid Angiography")
        print("="*60)

        print(f"\nField Strength: {info['field_strength']}")
        print(f"\nGeometry:")
        print(f"  FOV: {info['geometry']['fov']}")
        print(f"  Resolution: {info['geometry']['resolution']}")

        print(f"\nTiming:")
        print(f"  TR: {info['timing']['tr']}")
        print(f"  TE: {info['timing']['te']}")
        print(f"  Flip Angle: {info['timing']['flip_angle']}")

        print(f"\nReadout:")
        print(f"  Shots: {info['readout']['num_shots']}")
        print(f"  Samples/shot: {info['readout']['samples_per_shot']}")
        print(f"  Undersampling: {info['readout']['undersampling']}")
        print(f"  Trajectory: {info['readout']['trajectory']}")

        print(f"\nPreparation:")
        print(f"  T2prep: {info['preparation']['t2prep']}")
        print(f"  Inversion Time: {info['preparation']['inversion_time']}")
        print(f"  FatSat: {info['preparation']['fatsat']}")

        print(f"\nTissue Properties:")
        for tissue, props in info['tissue_properties'].items():
            print(f"  {tissue}: {props}")

        total_time = self.calculate_total_duration()
        print(f"\nTotal Duration: {total_time*1e3:.0f} ms")
        print("="*60 + "\n")


def create_boost_sequence(field_strength: float = 0.55) -> BOOSTSequence:
    """
    Factory function to create BOOST sequence.

    Args:
        field_strength: Magnetic field strength in Tesla (default 0.55)

    Returns:
        BOOSTSequence object ready for export
    """
    params = BOOSTParameters(field_strength=field_strength)
    return BOOSTSequence(params)


def export_boost_sequence(output_path: str = 'boost_carotid.seq') -> str:
    """
    Create and export BOOST sequence to file.

    Args:
        output_path: Path for output .seq file

    Returns:
        Path to exported file
    """
    boost = BOOSTSequence()
    boost.build_sequence()
    boost.export(output_path)
    boost.print_sequence_info()

    return output_path