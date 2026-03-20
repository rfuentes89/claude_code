"""
BOOST Sequence Implementation for Carotid Angiography at 0.55T

Bright-blood Optimized angiography using Single-shot Time-varied preparation
This package provides modular components for the BOOST sequence:

- Spiral trajectory with golden-angle ordering
- T2prep MLEV4 preparation
- Inversion pulse
- FatSat spectral saturation
- bSSFP readout
- Complete sequence assembly
- Bloch simulation for contrast verification

Reference:
    "BOOST: A Novel Bright-Blood Angiography Sequence" - ISMRM proceedings
"""

from .spiral_trajectory import SpiralTrajectory
from .t2prep import T2PrepMLEV4
from .inversion_pulse import InversionPulse
from .fat_sat import FatSatSpectral
from .bssfp_readout import bSSFPReadout
from .boost_sequence import BOOSTSequence

__all__ = [
    'SpiralTrajectory',
    'T2PrepMLEV4',
    'InversionPulse',
    'FatSatSpectral',
    'bSSFPReadout',
    'BOOSTSequence',
]

__version__ = '0.1.0'