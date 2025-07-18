#!/usr/bin/env python3
"""
Phase 1.4: Per-clip subtitles with word-timings
Generate SRT/VTT subtitles for each highlight segment with precise word-level timing
"""

import os
import sys
import json
import logging
import time
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SubtitleLine:
    """Single subtitle line with timing"""
    index: int
    start_ms: int
    end_ms: int
    text: str
    words: List[Dict[str, Any]]
    confidence: float

@dataclass
class SubtitleClip:
    """Subtitle data for a single highlight clip"""
    slug: str
    start_ms: int
    end_ms: int
    duration_ms: int
    lines: List[SubtitleLine]
    word_count: int
    total_confidence: float

class SubtitleGenerator:
    """Generate subtitles for highlight clips with word-level timing"""
    
    def __init__(self, max_chars_per_line: int = 42, max_lines_per_subtitle: int = 2, 
                 min_duration_ms: int = 1000, max_duration_ms: int = 6000):
        self.max_chars_per_line = max_chars_per_line
        self.max_lines_per_subtitle = max_lines_per_subtitle
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
        
    def format_timestamp_srt(self, ms: int) -> str:
        """Format timestamp for SRT format (HH:MM:SS,mmm)"""
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        milliseconds = ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    
    def format_timestamp_vtt(self, ms: int) -> str:
        """Format timestamp for VTT format (MM:SS.mmm)"""
        minutes = ms // 60000
        seconds = (ms % 60000) // 1000
        milliseconds = ms % 1000
        return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    
    def split_text_smartly(self, text: str, words: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
        """Split text into subtitle lines respecting word boundaries and timing"""
        lines = []
        current_line = ""
        current_words = []
        
        for word_data in words:
            word = word_data.get('word', '').strip()
            if not word:
                continue
                
            # Check if adding this word would exceed character limit
            test_line = current_line + (" " if current_line else "") + word
            
            if len(test_line) <= max_chars:
                # Add word to current line
                current_line = test_line
                current_words.append(word_data)
            else:
                # Start new line
                if current_line:
                    lines.append({
                        'text': current_line,
                        'words': current_words.copy(),
                        'start_ms': int(current_words[0]['start'] * 1000) if current_words else 0,
                        'end_ms': int(current_words[-1]['end'] * 1000) if current_words else 0
                    })
                
                current_line = word
                current_words = [word_data]
        
        # Add final line
        if current_line:
            lines.append({
                'text': current_line,
                'words': current_words.copy(),
                'start_ms': int(current_words[0]['start'] * 1000) if current_words else 0,
                'end_ms': int(current_words[-1]['end'] * 1000) if current_words else 0
            })
        
        return lines
    
    def optimize_subtitle_timing(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize subtitle timing for better readability"""
        optimized_lines = []
        
        for i, line in enumerate(lines):
            start_ms = line['start_ms']
            end_ms = line['end_ms']
            duration_ms = end_ms - start_ms
            
            # Ensure minimum duration
            if duration_ms < self.min_duration_ms:
                extension_needed = self.min_duration_ms - duration_ms
                # Extend end time by half the needed extension
                end_ms += extension_needed // 2
                # Extend start time backward by the other half (but not negative)
                start_ms = max(0, start_ms - extension_needed // 2)
            
            # Ensure maximum duration
            if duration_ms > self.max_duration_ms:
                # Reduce duration by cutting from the end
                end_ms = start_ms + self.max_duration_ms
            
            # Add small gap between subtitles for readability
            if i > 0:
                prev_end = optimized_lines[-1]['end_ms']
                gap_ms = start_ms - prev_end
                
                # Ensure minimum gap of 100ms
                if gap_ms < 100:
                    # Adjust start time to create gap
                    start_ms = prev_end + 100
                    # Adjust end time to maintain duration
                    end_ms = start_ms + min(duration_ms, self.max_duration_ms)
            
            optimized_lines.append({
                'text': line['text'],
                'words': line['words'],
                'start_ms': start_ms,
                'end_ms': end_ms
            })
        
        return optimized_lines
    
    def generate_clip_subtitles(self, highlight: Dict[str, Any], transcript_data: Dict[str, Any]) -> SubtitleClip:
        """Generate subtitles for a single highlight clip"""
        slug = highlight.get('slug', 'unknown')
        clip_start_ms = highlight.get('start_ms', 0)
        clip_end_ms = highlight.get('end_ms', 0)
        duration_ms = clip_end_ms - clip_start_ms
        
        logger.info(f"🎬 Generating subtitles for clip: {slug}")
        
        # Find all words within the highlight timeframe
        all_words = transcript_data.get('word_timings', [])
        clip_words = []
        
        for word_data in all_words:
            word_start_ms = int(word_data.get('start', 0) * 1000)
            word_end_ms = int(word_data.get('end', 0) * 1000)
            
            # Check if word overlaps with clip timeframe
            if word_start_ms < clip_end_ms and word_end_ms > clip_start_ms:
                # Adjust timing relative to clip start
                adjusted_word = word_data.copy()
                adjusted_word['start'] = max(0, (word_start_ms - clip_start_ms) / 1000.0)
                adjusted_word['end'] = min(duration_ms, (word_end_ms - clip_start_ms)) / 1000.0
                clip_words.append(adjusted_word)
        
        if not clip_words:
            logger.warning(f"No words found for clip {slug}")
            return SubtitleClip(
                slug=slug,
                start_ms=clip_start_ms,
                end_ms=clip_end_ms,
                duration_ms=duration_ms,
                lines=[],
                word_count=0,
                total_confidence=0.0
            )
        
        # Split text into subtitle lines
        full_text = " ".join([word.get('word', '').strip() for word in clip_words])
        lines_data = self.split_text_smartly(full_text, clip_words, self.max_chars_per_line)
        
        # Optimize timing
        optimized_lines = self.optimize_subtitle_timing(lines_data)
        
        # Create subtitle lines
        subtitle_lines = []
        total_confidence = 0.0
        
        for i, line_data in enumerate(optimized_lines):
            # Calculate average confidence for this line
            words_in_line = line_data['words']
            line_confidence = sum(word.get('confidence', 0.5) for word in words_in_line) / len(words_in_line) if words_in_line else 0.5
            total_confidence += line_confidence
            
            subtitle_line = SubtitleLine(
                index=i + 1,
                start_ms=line_data['start_ms'],
                end_ms=line_data['end_ms'],
                text=line_data['text'],
                words=words_in_line,
                confidence=line_confidence
            )
            subtitle_lines.append(subtitle_line)
        
        avg_confidence = total_confidence / len(subtitle_lines) if subtitle_lines else 0.0
        
        return SubtitleClip(
            slug=slug,
            start_ms=clip_start_ms,
            end_ms=clip_end_ms,
            duration_ms=duration_ms,
            lines=subtitle_lines,
            word_count=len(clip_words),
            total_confidence=avg_confidence
        )
    
    def generate_srt_content(self, subtitle_clip: SubtitleClip) -> str:
        """Generate SRT format content for a clip"""
        srt_content = []
        
        for line in subtitle_clip.lines:
            srt_content.append(str(line.index))
            srt_content.append(f"{self.format_timestamp_srt(line.start_ms)} --> {self.format_timestamp_srt(line.end_ms)}")
            srt_content.append(line.text)
            srt_content.append("")  # Empty line between subtitles
        
        return "\n".join(srt_content)
    
    def generate_vtt_content(self, subtitle_clip: SubtitleClip) -> str:
        """Generate VTT format content for a clip"""
        vtt_content = ["WEBVTT", ""]
        
        for line in subtitle_clip.lines:
            vtt_content.append(f"{self.format_timestamp_vtt(line.start_ms)} --> {self.format_timestamp_vtt(line.end_ms)}")
            vtt_content.append(line.text)
            vtt_content.append("")  # Empty line between subtitles
        
        return "\n".join(vtt_content)
    
    def generate_json_content(self, subtitle_clip: SubtitleClip) -> str:
        """Generate JSON format content for a clip with detailed word timings"""
        json_data = {
            "clip_info": {
                "slug": subtitle_clip.slug,
                "start_ms": subtitle_clip.start_ms,
                "end_ms": subtitle_clip.end_ms,
                "duration_ms": subtitle_clip.duration_ms,
                "word_count": subtitle_clip.word_count,
                "average_confidence": subtitle_clip.total_confidence
            },
            "subtitles": []
        }
        
        for line in subtitle_clip.lines:
            subtitle_data = {
                "index": line.index,
                "start_ms": line.start_ms,
                "end_ms": line.end_ms,
                "text": line.text,
                "confidence": line.confidence,
                "words": [
                    {
                        "word": word.get('word', ''),
                        "start": word.get('start', 0),
                        "end": word.get('end', 0),
                        "confidence": word.get('confidence', 0.5)
                    }
                    for word in line.words
                ]
            }
            json_data["subtitles"].append(subtitle_data)
        
        return json.dumps(json_data, indent=2)
    
    def save_subtitle_files(self, subtitle_clip: SubtitleClip, output_dir: str = "subtitles"):
        """Save subtitle files in multiple formats"""
        os.makedirs(output_dir, exist_ok=True)
        
        base_filename = f"{subtitle_clip.slug}"
        
        # Save SRT file
        srt_path = os.path.join(output_dir, f"{base_filename}.srt")
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_srt_content(subtitle_clip))
        
        # Save VTT file
        vtt_path = os.path.join(output_dir, f"{base_filename}.vtt")
        with open(vtt_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_vtt_content(subtitle_clip))
        
        # Save JSON file with word timings
        json_path = os.path.join(output_dir, f"{base_filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_json_content(subtitle_clip))
        
        logger.info(f"📄 Saved subtitles: {srt_path}, {vtt_path}, {json_path}")
        
        return {
            "srt": srt_path,
            "vtt": vtt_path,
            "json": json_path
        }

async def main():
    """Test subtitle generator"""
    if len(sys.argv) < 3:
        print("Usage: python phase1_4_subtitle_generator.py <transcript_json> <highlights_json>")
        return
    
    transcript_path = sys.argv[1]
    highlights_path = sys.argv[2]
    
    # Load data
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    with open(highlights_path, 'r') as f:
        highlights_data = json.load(f)
        highlights = highlights_data.get("top_highlights", [])
    
    # Initialize subtitle generator
    subtitle_gen = SubtitleGenerator()
    
    # Generate subtitles for each highlight
    subtitle_clips = []
    file_paths = []
    
    print("\n" + "=" * 60)
    print("🎬 SUBTITLE GENERATOR RESULTS")
    print("=" * 60)
    
    for i, highlight in enumerate(highlights[:5]):  # Top 5 highlights
        # Generate subtitles for this clip
        subtitle_clip = subtitle_gen.generate_clip_subtitles(highlight, transcript_data)
        subtitle_clips.append(subtitle_clip)
        
        # Save subtitle files
        paths = subtitle_gen.save_subtitle_files(subtitle_clip)
        file_paths.append(paths)
        
        # Print summary
        print(f"\n{i+1}. {subtitle_clip.slug}")
        print(f"   ⏱️  Duration: {subtitle_clip.duration_ms}ms")
        print(f"   📝 Lines: {len(subtitle_clip.lines)}")
        print(f"   🔤 Words: {subtitle_clip.word_count}")
        print(f"   🎯 Confidence: {subtitle_clip.total_confidence:.2f}")
        
        # Show first few subtitle lines
        for j, line in enumerate(subtitle_clip.lines[:2]):
            print(f"   {j+1}. [{line.start_ms}ms-{line.end_ms}ms] {line.text}")
    
    # Save master index
    master_index = {
        "generated_at": time.time(),
        "total_clips": len(subtitle_clips),
        "clips": [
            {
                "slug": clip.slug,
                "start_ms": clip.start_ms,
                "end_ms": clip.end_ms,
                "duration_ms": clip.duration_ms,
                "lines_count": len(clip.lines),
                "word_count": clip.word_count,
                "confidence": clip.total_confidence,
                "files": file_paths[i]
            }
            for i, clip in enumerate(subtitle_clips)
        ]
    }
    
    master_path = "subtitles/master_index.json"
    with open(master_path, 'w') as f:
        json.dump(master_index, f, indent=2)
    
    print(f"\n💾 Master index saved to: {master_path}")
    print(f"📁 All subtitle files saved to: subtitles/")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())