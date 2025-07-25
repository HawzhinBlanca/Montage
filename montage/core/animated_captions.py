#!/usr/bin/env python3
"""
Animated Captions with Word-Level Timing
Creates engaging, animated subtitles for vertical videos
"""
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import subprocess
import tempfile
import os

logger = logging.getLogger(__name__)


@dataclass
class WordTiming:
    """Word with precise timing"""
    word: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class CaptionSegment:
    """Caption segment with styling"""
    words: List[WordTiming]
    style: str = "default"
    animation: str = "fade"
    position: str = "bottom"
    speaker: Optional[str] = None


class AnimatedCaptionGenerator:
    """Generate animated captions with word-level timing"""
    
    def __init__(self):
        self.styles = {
            "default": {
                "font": "Helvetica",
                "size": 48,
                "color": "white",
                "stroke_color": "black",
                "stroke_width": 3,
                "background": "rgba(0,0,0,0.5)",
                "padding": 10
            },
            "emphasis": {
                "font": "Helvetica-Bold",
                "size": 56,
                "color": "yellow",
                "stroke_color": "black",
                "stroke_width": 4,
                "background": "rgba(0,0,0,0.7)",
                "padding": 15
            },
            "minimal": {
                "font": "Helvetica",
                "size": 42,
                "color": "white",
                "stroke_color": None,
                "stroke_width": 0,
                "background": None,
                "padding": 5
            }
        }
        
        self.animations = {
            "fade": "fade_in_out",
            "slide": "slide_up",
            "pop": "pop_in",
            "typewriter": "typewriter",
            "karaoke": "karaoke_highlight"
        }
        
    def generate_animated_captions(
        self,
        transcript: Dict[str, Any],
        video_path: str,
        output_path: str,
        style: str = "default",
        animation: str = "karaoke"
    ) -> bool:
        """
        Generate video with animated captions
        
        Args:
            transcript: Transcript with word timings
            video_path: Input video path
            output_path: Output video path
            style: Caption style
            animation: Animation type
        """
        # Convert transcript to caption segments
        segments = self._create_caption_segments(transcript)
        
        # Generate caption files
        ass_file = self._generate_ass_subtitles(segments, style, animation)
        
        # Apply captions to video
        success = self._apply_captions_to_video(
            video_path, ass_file, output_path
        )
        
        # Cleanup
        os.unlink(ass_file)
        
        return success
        
    def _create_caption_segments(self, transcript: Dict) -> List[CaptionSegment]:
        """Convert transcript to caption segments"""
        segments = []
        
        for segment in transcript.get("segments", []):
            words = []
            
            # Extract word timings
            for word_data in segment.get("words", []):
                words.append(WordTiming(
                    word=word_data["word"],
                    start=word_data["start"],
                    end=word_data["end"],
                    confidence=word_data.get("confidence", 1.0)
                ))
                
            if words:
                # Determine style based on content
                style = self._determine_style(segment.get("text", ""))
                
                segments.append(CaptionSegment(
                    words=words,
                    style=style,
                    animation="karaoke" if len(words) > 5 else "fade",
                    speaker=segment.get("speaker")
                ))
                
        return segments
        
    def _determine_style(self, text: str) -> str:
        """Determine caption style based on content"""
        text_lower = text.lower()
        
        # Emphasis for important moments
        emphasis_words = ["important", "remember", "key", "critical", "amazing"]
        if any(word in text_lower for word in emphasis_words):
            return "emphasis"
            
        # Minimal for fast speech
        if len(text.split()) > 15:
            return "minimal"
            
        return "default"
        
    def _generate_ass_subtitles(
        self,
        segments: List[CaptionSegment],
        style: str,
        animation: str
    ) -> str:
        """Generate Advanced SubStation Alpha format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ass', delete=False) as f:
            # Write ASS header
            f.write(self._create_ass_header(style))
            
            # Write dialogue events
            f.write("\n[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
            
            for segment in segments:
                if animation == "karaoke":
                    dialogue = self._create_karaoke_dialogue(segment)
                elif animation == "typewriter":
                    dialogue = self._create_typewriter_dialogue(segment)
                else:
                    dialogue = self._create_standard_dialogue(segment)
                    
                f.write(dialogue + "\n")
                
            return f.name
            
    def _create_ass_header(self, style_name: str) -> str:
        """Create ASS file header with styles"""
        style = self.styles[style_name]
        
        header = """[Script Info]
Title: Animated Captions
ScriptType: v4.00+
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
"""
        
        # Convert style to ASS format
        primary_color = self._color_to_ass(style["color"])
        outline_color = self._color_to_ass(style["stroke_color"]) if style["stroke_color"] else "&H00000000"
        
        style_line = f"Style: Default,{style['font']},{style['size']},{primary_color},&H00FFFFFF,{outline_color},&H00000000,0,0,0,0,100,100,0,0,1,{style['stroke_width']},0,2,10,10,10,1"
        
        header += style_line + "\n"
        
        # Add animation styles
        header += "Style: Emphasis,Helvetica-Bold,56,&H00FFFF00,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,4,0,2,10,10,10,1\n"
        
        return header
        
    def _color_to_ass(self, color: str) -> str:
        """Convert color name to ASS format"""
        colors = {
            "white": "&H00FFFFFF",
            "black": "&H00000000",
            "yellow": "&H00FFFF00",
            "red": "&H000000FF",
            "blue": "&H00FF0000",
            "green": "&H0000FF00"
        }
        return colors.get(color, "&H00FFFFFF")
        
    def _create_karaoke_dialogue(self, segment: CaptionSegment) -> str:
        """Create karaoke-style animated dialogue"""
        start_time = self._format_time(segment.words[0].start)
        end_time = self._format_time(segment.words[-1].end)
        
        # Build karaoke tags
        text = ""
        for i, word in enumerate(segment.words):
            duration = int((word.end - word.start) * 100)
            text += f"{{\\k{duration}}}{word.word}"
            if i < len(segment.words) - 1:
                text += " "
                
        return f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}"
        
    def _create_typewriter_dialogue(self, segment: CaptionSegment) -> str:
        """Create typewriter effect dialogue"""
        dialogues = []
        
        for i, word in enumerate(segment.words):
            start_time = self._format_time(word.start)
            end_time = self._format_time(segment.words[-1].end)
            
            # Show words progressively
            text = " ".join(w.word for w in segment.words[:i+1])
            
            dialogues.append(
                f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}"
            )
            
        return "\n".join(dialogues)
        
    def _create_standard_dialogue(self, segment: CaptionSegment) -> str:
        """Create standard dialogue with fade"""
        start_time = self._format_time(segment.words[0].start)
        end_time = self._format_time(segment.words[-1].end)
        
        text = " ".join(word.word for word in segment.words)
        
        # Add fade effect
        fade_duration = 20  # 0.2 seconds
        text = f"{{\\fad({fade_duration},{fade_duration})}}{text}"
        
        return f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}"
        
    def _format_time(self, seconds: float) -> str:
        """Format time for ASS format (h:mm:ss.cc)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        return f"{hours}:{minutes:02d}:{secs:05.2f}"
        
    def _apply_captions_to_video(
        self,
        video_path: str,
        ass_file: str,
        output_path: str
    ) -> bool:
        """Apply ASS subtitles to video using FFmpeg"""
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"ass={ass_file}",
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
            logger.error(f"Failed to apply captions: {e}")
            return False


class CreativeCaptionStyler:
    """Advanced caption styling based on content"""
    
    def __init__(self):
        self.emotion_styles = {
            "happy": {"color": "yellow", "animation": "pop"},
            "sad": {"color": "blue", "animation": "fade"},
            "angry": {"color": "red", "animation": "shake"},
            "excited": {"color": "orange", "animation": "bounce"},
            "calm": {"color": "green", "animation": "slide"}
        }
        
    def style_by_emotion(self, text: str, emotion: str) -> Dict[str, Any]:
        """Get style based on emotion"""
        base_style = self.emotion_styles.get(emotion, {
            "color": "white",
            "animation": "fade"
        })
        
        # Adjust for text length
        if len(text) > 50:
            base_style["size"] = 36  # Smaller for long text
        elif len(text) < 20:
            base_style["size"] = 56  # Larger for short text
            
        return base_style
        
    def create_dynamic_positioning(
        self,
        segments: List[CaptionSegment],
        avoid_faces: bool = True
    ) -> List[CaptionSegment]:
        """Dynamically position captions to avoid faces"""
        positioned = []
        
        for segment in segments:
            # Default positions
            positions = ["bottom", "top", "middle"]
            
            # Rotate through positions
            position = positions[len(positioned) % len(positions)]
            
            segment.position = position
            positioned.append(segment)
            
        return positioned


class MotionCaptionRenderer:
    """Render captions with motion graphics"""
    
    def __init__(self):
        self.motion_presets = {
            "slide_up": {
                "enter": "y:h+50",
                "exit": "y:-50",
                "duration": 0.3
            },
            "zoom_in": {
                "enter": "scale:0",
                "exit": "scale:1",
                "duration": 0.2
            },
            "rotate_in": {
                "enter": "angle:-180",
                "exit": "angle:0",
                "duration": 0.4
            }
        }
        
    def add_motion_blur(self, text: str, speed: float) -> str:
        """Add motion blur effect for fast animations"""
        if speed > 100:  # pixels per second
            return f"{{\\blur{int(speed/50)}}}{text}"
        return text