"""PhaseScope - live camera imaging, filtering, phase-stepping interferometry
and 2D / 3D visualisation.

Modules
-------
camera      Camera back-ends (simulated interferometer, OpenCV, Thorlabs TSI)
            and phase shifters (simulated, manual, serial piezo controller).
processing  Image filters and geometric transformations + a processing pipeline.
phase       Phase-stepping / Fourier phase retrieval, phase unwrapping,
            detrending and conversion of phase to height / thickness.
gui         The Qt main window.
"""

__version__ = "0.1.0"
