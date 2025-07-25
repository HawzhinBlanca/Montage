"""AI Creative Director - Smart Video Editing Brain"""
import json
import logging
from typing import List, Dict, Any
from ..vision.tracker import VisualTracker
from ..core.analyze_video import analyze_video

logger = logging.getLogger(__name__)

class AICreativeDirector:
    """AI-powered creative decision maker for video editing"""

    def __init__(self):
        self.visual_tracker = VisualTracker()
        self.highlight_threshold = 0.6
        self.max_highlights = 10

    def create_smart_edit(self, video_path: str, target_duration: int = 60) -> Dict:
        """Create intelligent video edit plan"""
        logger.info(f"AI Creative Director analyzing: {video_path}")

        # 1. Visual analysis
        visual_data = self.visual_tracker.analyze_video(video_path)

        # 2. Audio analysis (reuse existing)
        try:
            audio_analysis = analyze_video(video_path)
            transcript = audio_analysis.get("transcript", [])
        except Exception as e:
            logger.warning(f"Audio analysis failed: {e}")
            transcript = []

        # 3. Scene scoring
        scored_scenes = self.score_scenes(visual_data["scenes"], transcript)

        # 4. Intelligent selection
        highlights = self.select_highlights(scored_scenes, target_duration)

        # 5. Create edit plan
        edit_plan = self.generate_edit_plan(highlights, video_path)

        logger.info(f"Generated {len(highlights)} highlights for {target_duration}s video")
        return edit_plan

    def score_scenes(self, visual_scenes: List[Dict], transcript: List[Dict]) -> List[Dict]:
        """Score scenes using multi-modal analysis"""
        scored_scenes = []

        for scene in visual_scenes:
            # Visual interest score (already calculated)
            visual_score = scene["visual_interest_score"]

            # Audio relevance score
            audio_score = self.calculate_audio_score(scene["timestamp"], transcript)

            # Composition quality score
            composition_score = self.calculate_composition_score(scene["composition"])

            # Subject prominence score
            subject_score = self.calculate_subject_score(scene["primary_subjects"])

            # Composite score
            total_score = (
                visual_score * 0.3 +
                audio_score * 0.3 +
                composition_score * 0.2 +
                subject_score * 0.2
            )

            scene["scores"] = {
                "visual": visual_score,
                "audio": audio_score,
                "composition": composition_score,
                "subject": subject_score,
                "total": total_score
            }

            scored_scenes.append(scene)

        return scored_scenes

    def calculate_audio_score(self, timestamp: float, transcript: List[Dict]) -> float:
        """Score based on transcript content around timestamp"""
        if not transcript:
            return 0.3  # Neutral score if no transcript

        # Find transcript segments near this timestamp
        relevant_text = []
        for segment in transcript:
            if abs(segment.get("start", 0) - timestamp) < 2.0:  # Within 2 seconds
                relevant_text.append(segment.get("text", "").lower())

        if not relevant_text:
            return 0.3

        text = " ".join(relevant_text)

        # High-value keywords (can be configured)
        high_value_words = [
            "amazing", "incredible", "wow", "fantastic", "beautiful",
            "important", "key", "main", "first", "best", "great",
            "question", "answer", "explain", "show", "demonstrate"
        ]

        # Count high-value words
        word_score = sum(1 for word in high_value_words if word in text)

        # Normalize to 0-1 scale
        return min(word_score / 5, 1.0)

    def calculate_composition_score(self, composition: Dict) -> float:
        """Score based on visual composition quality"""
        brightness = composition["brightness"]
        contrast = composition["contrast"]

        # Prefer good lighting (not too dark/bright)
        brightness_score = 1.0 - abs(brightness - 128) / 128

        # Prefer good contrast (clarity)
        contrast_score = min(contrast / 80, 1.0)

        return (brightness_score + contrast_score) / 2

    def calculate_subject_score(self, subjects: List[Dict]) -> float:
        """Score based on subject presence and prominence"""
        if not subjects:
            return 0.2  # Low score for no subjects

        # More subjects = potentially more interesting
        subject_count_score = min(len(subjects) / 3, 1.0) * 0.5

        # Larger subjects = more prominent
        size_scores = [s.get("size_ratio", 0) for s in subjects]
        size_score = min(max(size_scores) if size_scores else 0, 1.0) * 0.5

        return subject_count_score + size_score

    def select_highlights(self, scored_scenes: List[Dict], target_duration: int) -> List[Dict]:
        """Intelligently select highlights for target duration"""
        # Sort by total score
        sorted_scenes = sorted(scored_scenes, key=lambda x: x["scores"]["total"], reverse=True)

        # Select top scenes that fit duration
        selected = []
        total_time = 0
        scene_duration = 3.0  # Default 3-second clips

        for scene in sorted_scenes:
            if total_time + scene_duration <= target_duration:
                scene["clip_duration"] = scene_duration
                scene["start_time"] = scene["timestamp"]
                scene["end_time"] = scene["timestamp"] + scene_duration
                selected.append(scene)
                total_time += scene_duration

            if len(selected) >= self.max_highlights:
                break

        # Sort selected clips chronologically
        return sorted(selected, key=lambda x: x["timestamp"])

    def generate_edit_plan(self, highlights: List[Dict], source_video: str) -> Dict:
        """Generate final edit plan in standard format"""
        clips = []

        for i, highlight in enumerate(highlights):
            clips.append({
                "id": f"highlight_{i}",
                "start_time": highlight["start_time"],
                "end_time": highlight["end_time"],
                "source": source_video,
                "scores": highlight["scores"],
                "reason": self.explain_selection(highlight)
            })

        return {
            "version": "2.0",
            "source_video": source_video,
            "clips": clips,
            "metadata": {
                "total_clips": len(clips),
                "total_duration": sum(c["end_time"] - c["start_time"] for c in clips),
                "ai_director": "v1.0",
                "selection_method": "multi_modal_scoring"
            }
        }

    def explain_selection(self, highlight: Dict) -> str:
        """Generate human-readable explanation for clip selection"""
        scores = highlight["scores"]
        reasons = []

        if scores["visual"] > 0.7:
            reasons.append("high visual interest")
        if scores["audio"] > 0.7:
            reasons.append("important dialogue")
        if scores["subject"] > 0.7:
            reasons.append("prominent subjects")
        if scores["composition"] > 0.8:
            reasons.append("excellent composition")

        if not reasons:
            reasons.append("balanced overall quality")

        return f"Selected for: {', '.join(reasons)} (score: {scores['total']:.2f})"
