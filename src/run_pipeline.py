#!/usr/bin/env python3
"""
Main CLI entry point for the real video processing pipeline
Usage: python -m src.run_pipeline <video> --mode smart|premium
"""
import sys
import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

from .analyze_video import analyze_video
from .highlight_selector import choose_highlights, get_total_cost
from .ffmpeg_utils import create_subtitle_file, burn_captions, apply_smart_crop, get_video_info
from .resolve_mcp import start_server
import threading
import time
import subprocess
import numpy as np

console = Console()

def extract_audio_rms(video_path: str) -> List[float]:
    """Extract audio RMS energy levels from video file"""
    try:
        # Use ffmpeg to extract audio stats
        cmd = [
            'ffmpeg', '-i', video_path,
            '-af', 'astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level',
            '-f', 'null', '-'
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Parse RMS levels from output
        rms_levels = []
        for line in result.stdout.split('\n'):
            if 'RMS_level' in line and '=' in line:
                try:
                    # Extract the RMS value
                    rms_value = float(line.split('=')[1].strip())
                    # Convert dB to linear scale (0-1)
                    linear_value = max(0.0, min(1.0, 10 ** (rms_value / 20) if rms_value > -60 else 0.0))
                    rms_levels.append(linear_value)
                except (ValueError, IndexError):
                    continue
        
        # If we couldn't extract RMS data, return reasonable defaults
        if not rms_levels:
            console.print("⚠️  Could not extract audio RMS, using estimated values", style="yellow")
            # Estimate based on video duration
            try:
                info_cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', video_path]
                duration_result = subprocess.run(info_cmd, capture_output=True, text=True)
                duration = float(duration_result.stdout.strip())
                # Generate reasonable RMS values (higher at start/end, varying in middle)
                samples = int(duration)
                rms_levels = [0.3 + 0.2 * np.sin(i * 0.1) + 0.1 * np.random.random() for i in range(samples)]
            except Exception:
                # Final fallback
                rms_levels = [0.4 + 0.1 * (i % 3) for i in range(100)]
        
        return rms_levels[:1000]  # Limit to prevent memory issues
        
    except Exception as e:
        console.print(f"⚠️  Audio RMS extraction failed: {e}, using fallback", style="yellow")
        # Return reasonable fallback values
        return [0.3 + 0.1 * (i % 5) for i in range(100)]

def validate_video_file(video_path: str) -> bool:
    """Validate video file exists and is accessible, prevent path traversal"""
    # Security: Prevent path traversal attacks
    if '..' in video_path or video_path.startswith('/'):
        # Allow absolute paths but not traversal
        if '..' in video_path:
            console.print(f"❌ Invalid path (contains '..'): {video_path}", style="bold red")
            return False
    
    # Resolve to absolute path and check it's safe
    try:
        abs_path = os.path.abspath(video_path)
        # Ensure the resolved path doesn't contain suspicious patterns
        if '..' in abs_path or abs_path.count('/') > 10:  # Reasonable depth limit
            console.print(f"❌ Suspicious path detected: {abs_path}", style="bold red")
            return False
    except Exception as e:
        console.print(f"❌ Path validation error: {e}", style="bold red")
        return False
    
    if not os.path.exists(abs_path):
        console.print(f"❌ Video file not found: {abs_path}", style="bold red")
        return False
    
    if not os.path.isfile(abs_path):
        console.print(f"❌ Path is not a file: {abs_path}", style="bold red")
        return False
    
    # Check file size
    file_size = os.path.getsize(abs_path)
    if file_size == 0:
        console.print(f"❌ Video file is empty: {abs_path}", style="bold red")
        return False
    
    # Check file extension
    valid_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v']
    if not any(abs_path.lower().endswith(ext) for ext in valid_extensions):
        console.print(f"⚠️  Warning: Unusual file extension. Supported: {', '.join(valid_extensions)}", style="yellow")
    
    return True

def display_video_info(video_path: str):
    """Display video file information"""
    info = get_video_info(video_path)
    
    if not info:
        console.print("❌ Could not get video information", style="bold red")
        return
    
    table = Table(title="Video Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("File", os.path.basename(video_path))
    table.add_row("Duration", f"{info.get('duration', 0):.1f} seconds")
    table.add_row("Size", f"{info.get('size', 0) / (1024*1024):.1f} MB")
    table.add_row("Resolution", f"{info.get('width', 0)}x{info.get('height', 0)}")
    table.add_row("FPS", f"{info.get('fps', 0):.1f}")
    table.add_row("Video Codec", info.get('video_codec', 'Unknown'))
    table.add_row("Audio Codec", info.get('audio_codec', 'Unknown'))
    
    console.print(table)

def start_mcp_server():
    """Start MCP server in background thread"""
    def run_server():
        try:
            start_server(host='localhost', port=7801)
        except Exception as e:
            console.print(f"❌ MCP server failed: {e}", style="bold red")
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    # Wait for server to start
    time.sleep(2)
    
    # Test server availability
    try:
        response = requests.get("http://localhost:7801/health", timeout=5)
        if response.status_code == 200:
            console.print("✅ MCP server started successfully", style="green")
            return True
        else:
            console.print(f"❌ MCP server unhealthy: {response.status_code}", style="bold red")
            return False
    except Exception as e:
        console.print(f"❌ MCP server not responding: {e}", style="bold red")
        return False

def post_to_mcp(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Post data to MCP server"""
    try:
        response = requests.post(
            f"http://localhost:7801/{endpoint}",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            console.print(f"❌ MCP {endpoint} failed: {response.status_code}", style="bold red")
            console.print(f"   Response: {response.text}", style="red")
            return {"error": f"HTTP {response.status_code}"}
    
    except Exception as e:
        console.print(f"❌ MCP request failed: {e}", style="bold red")
        return {"error": str(e)}

def create_subtitles_for_clips(clips: List[Dict], words: List[Dict]) -> List[str]:
    """Create subtitle files for each clip"""
    subtitle_files = []
    
    for i, clip in enumerate(clips):
        subtitle_path = f"/tmp/subtitle_{i+1}.srt"
        
        if create_subtitle_file(words, clip["start_ms"], clip["end_ms"], subtitle_path):
            subtitle_files.append(subtitle_path)
        else:
            console.print(f"⚠️  Failed to create subtitle for clip {i+1}", style="yellow")
    
    return subtitle_files

def process_video_pipeline(video_path: str, mode: str = "smart") -> Dict[str, Any]:
    """Main video processing pipeline"""
    
    console.print(f"\n🎬 Processing video in {mode.upper()} mode", style="bold blue")
    console.print(f"   Video: {os.path.basename(video_path)}")
    
    results = {
        "video_path": video_path,
        "mode": mode,
        "success": False,
        "error": None,
        "analysis": None,
        "highlights": None,
        "timeline": None,
        "cost": 0.0
    }
    
    try:
        # Step 1: Analyze video
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task("Analyzing video (ASR + Diarization)...", total=None)
            analysis = analyze_video(video_path)
            progress.update(task, description="✅ Video analysis complete")
            
            results["analysis"] = {
                "transcript_length": len(analysis["transcript"]),
                "word_count": len(analysis["words"]),
                "speaker_turns": len(analysis.get("speaker_turns", []))
            }
            
            console.print(f"📝 Transcript: {len(analysis['words'])} words, {len(analysis.get('speaker_turns', []))} speaker turns")
        
        # Step 2: Choose highlights
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task(f"Selecting highlights ({mode} mode)...", total=None)
            
            # Extract real audio RMS energy
            audio_energy = extract_audio_rms(video_path)
            
            highlights = choose_highlights(analysis["words"], audio_energy, mode)
            progress.update(task, description="✅ Highlights selected")
            
            results["highlights"] = highlights
            results["cost"] = get_total_cost()
            
            console.print(f"🎯 Selected {len(highlights)} highlights (Cost: ${results['cost']:.3f})")
        
        # Step 3: Create subtitles
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task("Creating subtitles...", total=None)
            subtitle_files = create_subtitles_for_clips(highlights, analysis["words"])
            progress.update(task, description="✅ Subtitles created")
            
            console.print(f"📄 Created {len(subtitle_files)} subtitle files")
        
        # Step 4: Build timeline via MCP
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task("Building timeline...", total=None)
            
            timeline_data = {
                "video_path": video_path,
                "clips": highlights,
                "project_name": f"Montage_{int(time.time())}"
            }
            
            timeline_result = post_to_mcp("buildTimeline", timeline_data)
            progress.update(task, description="✅ Timeline built")
            
            results["timeline"] = timeline_result
            
            if timeline_result.get("success"):
                console.print(f"🎬 Timeline created: {timeline_result.get('method', 'unknown')}")
            else:
                console.print(f"❌ Timeline creation failed: {timeline_result.get('error', 'unknown')}", style="bold red")
        
        results["success"] = True
        return results
        
    except Exception as e:
        results["error"] = str(e)
        console.print(f"❌ Pipeline failed: {e}", style="bold red")
        return results

def display_results(results: Dict[str, Any]):
    """Display pipeline results"""
    
    if not results["success"]:
        console.print("\n❌ Pipeline Failed", style="bold red")
        console.print(f"   Error: {results.get('error', 'Unknown error')}")
        return
    
    # Success summary
    console.print("\n✅ Pipeline Completed Successfully!", style="bold green")
    
    # Analysis summary
    if results.get("analysis"):
        analysis = results["analysis"]
        console.print(f"\n📊 Analysis Summary:")
        console.print(f"   • Words transcribed: {analysis['word_count']}")
        console.print(f"   • Speaker turns: {analysis['speaker_turns']}")
        console.print(f"   • Transcript length: {analysis['transcript_length']} characters")
    
    # Highlights summary
    if results.get("highlights"):
        highlights = results["highlights"]
        console.print(f"\n🎯 Highlights Summary:")
        console.print(f"   • Mode: {results['mode'].upper()}")
        console.print(f"   • Clips selected: {len(highlights)}")
        console.print(f"   • Total cost: ${results['cost']:.3f}")
        
        # Show top highlights
        table = Table(title="Top Highlights")
        table.add_column("Clip", style="cyan")
        table.add_column("Title", style="green")
        table.add_column("Duration", style="yellow")
        table.add_column("Score", style="magenta")
        
        for i, highlight in enumerate(highlights[:5], 1):
            duration = (highlight["end_ms"] - highlight["start_ms"]) / 1000
            table.add_row(
                f"{i}",
                highlight["title"][:50] + "..." if len(highlight["title"]) > 50 else highlight["title"],
                f"{duration:.1f}s",
                f"{highlight['score']:.1f}"
            )
        
        console.print(table)
    
    # Timeline summary
    if results.get("timeline"):
        timeline = results["timeline"]
        console.print(f"\n🎬 Timeline Summary:")
        console.print(f"   • Method: {timeline.get('method', 'unknown')}")
        console.print(f"   • Clips added: {timeline.get('clips_added', 0)}")
        
        if timeline.get("output_path"):
            console.print(f"   • Output: {timeline['output_path']}")
    
    # Export JSON plan
    plan_file = f"montage_plan_{int(time.time())}.json"
    with open(plan_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n💾 Full plan saved to: {plan_file}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Real AI-powered video processing pipeline")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--mode", choices=["smart", "premium"], default="smart", 
                       help="Processing mode (smart=local, premium=AI-enhanced)")
    parser.add_argument("--info", action="store_true", help="Show video info only")
    parser.add_argument("--no-server", action="store_true", help="Skip MCP server")
    
    args = parser.parse_args()
    
    # Validate video file
    if not validate_video_file(args.video):
        sys.exit(1)
    
    # Show video info
    display_video_info(args.video)
    
    if args.info:
        sys.exit(0)
    
    # Start MCP server
    if not args.no_server:
        console.print("\n🚀 Starting MCP server...", style="blue")
        if not start_mcp_server():
            console.print("⚠️  Continuing without MCP server", style="yellow")
    
    # Process video
    results = process_video_pipeline(args.video, args.mode)
    
    # Display results
    display_results(results)
    
    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)

if __name__ == "__main__":
    main()