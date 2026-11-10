"""ATAN-DN: telemetry anomaly detection with Bounded-Influence Gain Normalization."""
__version__ = "0.1.0"

try:  # tolerate a partially populated package during development
    from .layers.bign import BIGN1D, ExactInverseBIGN1D, MirrorBIGN1D
    from .losses import rr_wfl_loss
    from .models.atandn import ATANDN
except ImportError:  # pragma: no cover
    pass

__all__ = ["__version__", "ATANDN", "BIGN1D", "ExactInverseBIGN1D", "MirrorBIGN1D", "rr_wfl_loss"]
