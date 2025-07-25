# src/compat.py
import warnings

from .core.beats import *  # noqa: F403

warnings.warn("Import from montage.core.beats; compat will be removed in Q4-2025", DeprecationWarning)
