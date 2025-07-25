#!/usr/bin/env python3
"""
Intelligent Emoji Overlay System
AI-driven emoji placement and animation
"""
import os
import json
import logging
import subprocess
import tempfile
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EmojiAnimation(Enum):
    """Emoji animation types"""
    FADE = "fade"
    POP = "pop"
    SLIDE = "slide"
    BOUNCE = "bounce"
    ROTATE = "rotate"
    SCALE_BOUNCE = "scale_bounce"
    FLOAT = "float"
    EXPLODE = "explode"


@dataclass
class EmojiOverlay:
    """Represents an emoji overlay"""
    emoji: str
    start_time: float
    duration: float
    position: Tuple[int, int]  # (x, y)
    size: int
    animation: EmojiAnimation
    animation_params: Dict[str, Any]


class IntelligentEmojiSystem:
    """AI-driven emoji overlay system"""
    
    def __init__(self):
        # Emotion to emoji mapping
        self.emotion_emojis = {
            "happy": ["😊", "😄", "🎉", "✨", "🌟"],
            "sad": ["😢", "😔", "💔", "😭", "☔"],
            "angry": ["😠", "😡", "🔥", "💢", "😤"],
            "surprised": ["😱", "😮", "🤯", "😲", "⚡"],
            "love": ["❤️", "💕", "😍", "💖", "💝"],
            "thinking": ["🤔", "💭", "🧐", "❓", "💡"],
            "funny": ["😂", "🤣", "😆", "💀", "🤡"],
            "cool": ["😎", "🔥", "💯", "⚡", "🚀"],
            "celebration": ["🎉", "🎊", "🥳", "🎈", "🍾"],
            "warning": ["⚠️", "🚨", "❗", "⛔", "🔴"]
        }
        
        # Context-based emoji triggers
        self.context_triggers = {
            "money": ["💰", "💵", "💸", "🤑", "💳"],
            "food": ["🍕", "🍔", "🍟", "🥗", "🍰"],
            "travel": ["✈️", "🌍", "🗺️", "🏖️", "🎒"],
            "tech": ["💻", "📱", "🤖", "⚡", "🔧"],
            "music": ["🎵", "🎶", "🎤", "🎧", "🎸"],
            "sports": ["⚽", "🏀", "🎾", "🏆", "💪"],
            "education": ["📚", "🎓", "✏️", "🧑‍🎓", "💡"],
            "nature": ["🌳", "🌺", "🌈", "☀️", "🌊"]
        }
        
    def analyze_and_place_emojis(
        self,
        transcript: Dict[str, Any],
        emotions: List[Dict[str, Any]],
        video_dimensions: Tuple[int, int]
    ) -> List[EmojiOverlay]:
        """
        Analyze content and place emojis intelligently
        
        Args:
            transcript: Video transcript with timings
            emotions: Emotion analysis results
            video_dimensions: (width, height) of video
            
        Returns:
            List of emoji overlays
        """
        overlays = []
        
        # Analyze each segment
        for i, segment in enumerate(transcript.get("segments", [])):
            text = segment["text"]
            start_time = segment["start"]
            
            # Get emotion for this segment
            segment_emotion = self._get_segment_emotion(emotions, start_time)
            
            # Detect context
            context = self._detect_context(text)
            
            # Select appropriate emojis
            emojis = self._select_emojis(text, segment_emotion, context)
            
            # Create overlays with smart positioning
            for emoji in emojis[:2]:  # Max 2 emojis per segment
                overlay = self._create_emoji_overlay(
                    emoji, start_time, video_dimensions, i
                )
                overlays.append(overlay)
                
        # Add reaction emojis at key moments
        overlays.extend(self._add_reaction_emojis(transcript, emotions))
        
        return overlays
        
    def _get_segment_emotion(self, emotions: List[Dict], timestamp: float) -> str:
        """Get emotion at specific timestamp"""
        for emotion in emotions:
            if emotion["start"] <= timestamp <= emotion["end"]:
                return emotion["emotion"]
        return "neutral"
        
    def _detect_context(self, text: str) -> List[str]:
        """Detect context from text"""
        text_lower = text.lower()
        detected_contexts = []
        
        for context, keywords in self.context_triggers.items():
            # Simple keyword matching (could use NLP)
            context_words = {
                "money": ["money", "cash", "dollar", "pay", "cost", "price"],
                "food": ["eat", "food", "hungry", "delicious", "cook", "meal"],
                "travel": ["travel", "trip", "visit", "fly", "vacation", "journey"],
                "tech": ["computer", "app", "software", "ai", "tech", "digital"],
                "music": ["music", "song", "sing", "dance", "melody", "beat"],
                "sports": ["game", "play", "win", "team", "score", "match"],
                "education": ["learn", "study", "teach", "school", "knowledge"],
                "nature": ["nature", "tree", "flower", "mountain", "ocean", "forest"]
            }
            
            if any(word in text_lower for word in context_words.get(context, [])):
                detected_contexts.append(context)
                
        return detected_contexts
        
    def _select_emojis(self, text: str, emotion: str, contexts: List[str]) -> List[str]:
        """Select appropriate emojis based on analysis"""
        selected = []
        
        # Add emotion-based emoji
        if emotion in self.emotion_emojis:
            selected.extend(self.emotion_emojis[emotion][:1])
            
        # Add context-based emojis
        for context in contexts:
            if context in self.context_triggers:
                selected.extend(self.context_triggers[context][:1])
                
        # Special triggers
        if "!" in text:
            selected.append("❗")
        if "?" in text:
            selected.append("❓")
        if any(word in text.lower() for word in ["amazing", "awesome", "incredible"]):
            selected.append("🤩")
            
        return list(dict.fromkeys(selected))  # Remove duplicates
        
    def _create_emoji_overlay(
        self,
        emoji: str,
        start_time: float,
        video_dims: Tuple[int, int],
        segment_index: int
    ) -> EmojiOverlay:
        """Create emoji overlay with smart positioning"""
        width, height = video_dims
        
        # Position based on segment index to avoid overlap
        positions = [
            (width * 0.8, height * 0.2),   # Top right
            (width * 0.2, height * 0.2),   # Top left
            (width * 0.5, height * 0.15),  # Top center
            (width * 0.8, height * 0.5),   # Middle right
            (width * 0.2, height * 0.5),   # Middle left
        ]
        
        position = positions[segment_index % len(positions)]
        
        # Select animation based on emoji type
        if emoji in ["🎉", "🎊", "🥳"]:
            animation = EmojiAnimation.EXPLODE
        elif emoji in ["❤️", "💕", "😍"]:
            animation = EmojiAnimation.FLOAT
        elif emoji in ["😱", "😮", "🤯"]:
            animation = EmojiAnimation.POP
        elif emoji in ["🔥", "💯", "⚡"]:
            animation = EmojiAnimation.SCALE_BOUNCE
        else:
            animation = EmojiAnimation.FADE
            
        return EmojiOverlay(
            emoji=emoji,
            start_time=start_time,
            duration=2.0,
            position=(int(position[0]), int(position[1])),
            size=80,
            animation=animation,
            animation_params=self._get_animation_params(animation)
        )
        
    def _get_animation_params(self, animation: EmojiAnimation) -> Dict[str, Any]:
        """Get parameters for animation type"""
        params = {
            EmojiAnimation.FADE: {
                "fade_in": 0.3,
                "fade_out": 0.3
            },
            EmojiAnimation.POP: {
                "scale_from": 0,
                "scale_to": 1.2,
                "settle_scale": 1.0,
                "pop_duration": 0.3
            },
            EmojiAnimation.BOUNCE: {
                "bounce_height": 20,
                "bounce_count": 3,
                "bounce_decay": 0.7
            },
            EmojiAnimation.FLOAT: {
                "float_distance": 50,
                "float_speed": 0.5,
                "opacity_fade": True
            },
            EmojiAnimation.EXPLODE: {
                "particle_count": 8,
                "explosion_radius": 100,
                "fade_duration": 0.5
            }
        }
        
        return params.get(animation, {})
        
    def _add_reaction_emojis(
        self,
        transcript: Dict,
        emotions: List[Dict]
    ) -> List[EmojiOverlay]:
        """Add reaction emojis at emotional peaks"""
        reaction_overlays = []
        
        # Find emotional peaks
        emotion_peaks = self._find_emotion_peaks(emotions)
        
        for peak in emotion_peaks:
            if peak["intensity"] > 0.8:
                # Strong emotional moment
                emoji = self._get_reaction_emoji(peak["emotion"])
                
                overlay = EmojiOverlay(
                    emoji=emoji,
                    start_time=peak["time"],
                    duration=1.5,
                    position=(540, 960),  # Center for 1080x1920
                    size=120,  # Larger for impact
                    animation=EmojiAnimation.EXPLODE,
                    animation_params={
                        "particle_count": 12,
                        "explosion_radius": 150,
                        "fade_duration": 0.7
                    }
                )
                
                reaction_overlays.append(overlay)
                
        return reaction_overlays
        
    def _find_emotion_peaks(self, emotions: List[Dict]) -> List[Dict]:
        """Find peaks in emotional intensity"""
        peaks = []
        
        for i in range(1, len(emotions) - 1):
            prev_intensity = emotions[i-1].get("intensity", 0.5)
            curr_intensity = emotions[i].get("intensity", 0.5)
            next_intensity = emotions[i+1].get("intensity", 0.5)
            
            # Local maximum
            if curr_intensity > prev_intensity and curr_intensity > next_intensity:
                peaks.append({
                    "time": emotions[i]["start"],
                    "emotion": emotions[i]["emotion"],
                    "intensity": curr_intensity
                })
                
        return peaks
        
    def _get_reaction_emoji(self, emotion: str) -> str:
        """Get reaction emoji for emotion peak"""
        reactions = {
            "happy": "🎉",
            "sad": "😢",
            "angry": "😡",
            "surprised": "🤯",
            "love": "💖",
            "funny": "😂"
        }
        return reactions.get(emotion, "✨")


class EmojiRenderer:
    """Render emojis on video"""
    
    def __init__(self):
        self.ffmpeg_path = "ffmpeg"
        
    def apply_emoji_overlays(
        self,
        video_path: str,
        output_path: str,
        overlays: List[EmojiOverlay]
    ) -> bool:
        """Apply emoji overlays to video"""
        # Create filter complex for all overlays
        filter_parts = []
        
        for i, overlay in enumerate(overlays):
            filter_parts.append(
                self._create_emoji_filter(overlay, i)
            )
            
        if not filter_parts:
            # No overlays to apply
            return False
            
        # Combine all filters
        filter_complex = ";".join(filter_parts)
        
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-filter_complex", filter_complex,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "copy",
            output_path,
            "-y"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply emoji overlays: {e}")
            return False
            
    def _create_emoji_filter(self, overlay: EmojiOverlay, index: int) -> str:
        """Create FFmpeg filter for emoji overlay"""
        x, y = overlay.position
        
        # Base filter with emoji text
        base_filter = (
            f"drawtext="
            f"text='{overlay.emoji}':"
            f"fontsize={overlay.size}:"
            f"x={x}:"
            f"y={y}:"
            f"enable='between(t,{overlay.start_time},{overlay.start_time + overlay.duration})'"
        )
        
        # Add animation
        if overlay.animation == EmojiAnimation.FADE:
            # Fade in/out
            params = overlay.animation_params
            fade_in = params.get("fade_in", 0.3)
            fade_out = params.get("fade_out", 0.3)
            
            base_filter += (
                f":alpha='if(lt(t,{overlay.start_time + fade_in}),"
                f"(t-{overlay.start_time})/{fade_in},"
                f"if(gt(t,{overlay.start_time + overlay.duration - fade_out}),"
                f"({overlay.start_time + overlay.duration}-t)/{fade_out},1))'"
            )
            
        elif overlay.animation == EmojiAnimation.BOUNCE:
            # Bounce animation
            base_filter = base_filter.replace(
                f"y={y}",
                f"y={y}-20*abs(sin(2*PI*(t-{overlay.start_time})*2))"
            )
            
        return base_filter