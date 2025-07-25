import json
import subprocess
import shlex
from pathlib import Path
from typing import Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Back-compat: enum used by earlier tests expecting predefined loudness targets
# ---------------------------------------------------------------------------


class NormalizationTarget(Enum):
    """Target loudness profiles (integrated LUFS, true-peak dBFS, loudness range)."""

    BROADCAST = (-23.0, -1.0, 7.0)
    LOUD = (-14.0, -1.0, 9.0)

    @property
    def i(self) -> float:  # Integrated loudness (LUFS)
        return self.value[0]

    @property
    def tp(self) -> float:  # True-peak (dBFS)
        return self.value[1]

    @property
    def lra(self) -> float:  # Loudness range (LU)
        return self.value[2]


class AudioNormalizer:
    def __init__(self, target_i: float = -23.0, target_tp: float = -1.0, target_lra: float = 7.0):
        self.target_i = target_i
        self.target_tp = target_tp
        self.target_lra = target_lra
        self.last_analysis: Optional[dict] = None

    def analyze(self, input_path: str) -> dict:
        if not Path(input_path).is_file():
            raise FileNotFoundError(f"Audio not found: {input_path}")
        cmd = [
            "ffmpeg", "-i", input_path,
            "-af", "loudnorm=print_format=json",
            "-f", "null", "-"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # extract JSON between braces
        jstart = res.stdout.find("{")
        jend = res.stdout.rfind("}") + 1
        data = json.loads(res.stdout[jstart:jend])
        self.last_analysis = data
        return data

    def normalize(self, input_path: str, output_path: str) -> None:
        data = self.analyze(input_path)
        measured_i = data["input_i"]
        measured_tp = data["input_tp"]
        measured_lra = data["input_lra"]
        measured_thresh = data["input_thresh"]
        offset = data["target_offset"]

        filter_str = (
            f"loudnorm=I={self.target_i}:TP={self.target_tp}:LRA={self.target_lra}:"
            f"measured_I={measured_i}:measured_TP={measured_tp}:"
            f"measured_LRA={measured_lra}:measured_thresh={measured_thresh}:"
            f"offset={offset}:linear=true:print_format=json"
        )
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", filter_str,
            "-c:a", "aac",
            output_path
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)


# Public API
__all__ = [
    "AudioNormalizer",
    "NormalizationTarget",
]
