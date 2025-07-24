import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict

SCENE_THRESHOLD = 30.0  # adjust per your content

def extract_frames(video_path: str, interval: float = 1.0) -> List[np.ndarray]:
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(int(fps * interval), 1)

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames

def detect_scene_changes(frames: List[np.ndarray]) -> List[int]:
    if not frames:
        return []
    scene_idxs = [0]
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i, frame in enumerate(frames[1:], start=1):
        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev, curr)
        if float(np.mean(diff)) > SCENE_THRESHOLD:
            scene_idxs.append(i)
        prev = curr
    return scene_idxs

def analyze_motion(frame1: np.ndarray, frame2: np.ndarray) -> float:
    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return float(np.mean(np.abs(flow)))

def analyze_video_content(video_path: str) -> Dict:
    frames = extract_frames(video_path)
    scenes = detect_scene_changes(frames)
    motions: List[float] = []
    for a, b in zip(frames, frames[1:]):
        motions.append(analyze_motion(a, b))
    avg_motion = float(np.mean(motions)) if motions else 0.0
    return {
        "frame_count": len(frames),
        "scene_changes": scenes,
        "average_motion": avg_motion
    }

def analyze_video(video_path: str, use_premium: bool = False) -> Dict:
    """Wrapper for backward compatibility with celery tasks"""
    content = analyze_video_content(video_path)
    # Generate simple highlights based on scene changes
    highlights = []
    for i, scene_idx in enumerate(content["scene_changes"][:5]):  # Max 5 highlights
        highlights.append({
            "start": float(scene_idx),
            "end": float(scene_idx + 5.0),
            "score": 0.8 - (i * 0.1),  # Decreasing scores
            "type": "scene_change"
        })
    return {
        "highlights": highlights,
        "analysis": content
    }
