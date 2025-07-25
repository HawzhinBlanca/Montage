import subprocess
import shlex
from pathlib import Path
from typing import List

class ConcatEditor:
    @staticmethod
    def concat(segment_paths: List[str], output_path: str) -> None:
        for seg in segment_paths:
            if not Path(seg).is_file():
                raise FileNotFoundError(f"Segment missing: {seg}")

        # write file list
        list_file = Path(output_path).with_suffix(".txt")
        with open(list_file, "w") as f:
            for seg in segment_paths:
                f.write(f"file '{seg}'\n")

        cmd = (
            f"ffmpeg -y -f concat -safe 0 "
            f"-i {shlex.quote(str(list_file))} "
            f"-c copy {shlex.quote(output_path)}"
        )
        subprocess.run(shlex.split(cmd), check=True)
