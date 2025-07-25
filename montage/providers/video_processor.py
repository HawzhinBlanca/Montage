import json
import subprocess
import shlex
import tempfile
from pathlib import Path
from typing import List, Dict, Any

def probe_codecs(video_path: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_streams", "-show_format",
        video_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(res.stdout)
    codecs = {
        s["codec_type"]: s.get("codec_name")
        for s in info.get("streams", [])
    }
    return codecs

class VideoEditor:
    def __init__(self, source_path: str):
        if not Path(source_path).is_file():
            raise FileNotFoundError(f"Source not found: {source_path}")
        self.source = source_path
        self.codecs = probe_codecs(source_path)

    def process_clips(self, clips: List[Dict], output_path: str) -> None:
        """
        clips: [{ "start": float, "end": float, "filters": Optional[str] }]
        """
        tmp_dir = Path(tempfile.mkdtemp(prefix="montage_"))
        segment_files = []
        for i, clip in enumerate(clips):
            start = clip["start"]
            duration = clip["end"] - start
            seg_path = tmp_dir / f"seg_{i:03d}.mp4"
            cmd = (
                f"ffmpeg -y -ss {start} -i {shlex.quote(self.source)} "
                f"-t {duration} "
            )
            vf = clip.get("filters")
            if vf:
                cmd += f"-vf {shlex.quote(vf)} "
            # preserve codec if supported
            vcodec = self.codecs.get("video", "libx264")
            acodec = self.codecs.get("audio", "aac")
            cmd += f"-c:v {vcodec} -c:a {acodec} {shlex.quote(str(seg_path))}"
            subprocess.run(shlex.split(cmd), check=True)
            segment_files.append(str(seg_path))

        # Create file list for concat
        list_txt = tmp_dir / "segments.txt"
        with open(list_txt, "w") as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")

        # Concat demuxer
        cmd = (
            f"ffmpeg -y -f concat -safe 0 -i {shlex.quote(str(list_txt))} "
            f"-c copy {shlex.quote(output_path)}"
        )
        subprocess.run(shlex.split(cmd), check=True)
