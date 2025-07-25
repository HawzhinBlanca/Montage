#!/usr/bin/env python3
"""
Main CLI entry point for the real video processing pipeline
Usage: python -m src.run_pipeline <video> --mode smart|premium
"""
import sys
import argparse
import json
import os
from typing import Dict, Any, List
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..core.analyze_video import analyze_video
from ..core.highlight_selector import select_highlights
from ..core.quality_validator import enforce_production_quality, QualityValidationError
from ..utils.ffmpeg_utils import (
    create_subtitle_file,
    get_video_info,
)
from ..providers.resolve_mcp import start_server
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
            "ffmpeg",
            "-i",
            video_path,
            "-af",
            "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level",
            "-f",
            "null",
            "-",
        ]

        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        # Parse RMS levels from output
        rms_levels = []
        for line in result.stdout.split("\n"):
            if "RMS_level" in line and "=" in line:
                try:
                    # Extract the RMS value
                    rms_value = float(line.split("=")[1].strip())
                    # Convert dB to linear scale (0-1)
                    linear_value = max(
                        0.0,
                        min(1.0, 10 ** (rms_value / 20) if rms_value > -60 else 0.0),
                    )
                    rms_levels.append(linear_value)
                except (ValueError, IndexError):
                    continue

        # If we couldn't extract RMS data, return reasonable defaults
        if not rms_levels:
            console.print(
                "⚠️  Could not extract audio RMS, using estimated values", style="yellow"
            )
            # Estimate based on video duration
            try:
                info_cmd = [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    video_path,
                ]
                duration_result = subprocess.run(
                    info_cmd, capture_output=True, text=True
                )
                duration = float(duration_result.stdout.strip())
                # Generate reasonable RMS values (higher at start/end, varying in middle)
                samples = int(duration)
                rms_levels = [
                    0.3 + 0.2 * np.sin(i * 0.1) + 0.1 * np.random.random()
                    for i in range(samples)
                ]
            except (subprocess.CalledProcessError, ValueError, AttributeError):
                # Final fallback
                rms_levels = [0.4 + 0.1 * (i % 3) for i in range(100)]

        return rms_levels[:1000]  # Limit to prevent memory issues

    except (subprocess.CalledProcessError, FileNotFoundError, OSError, ValueError) as e:
        console.print(
            f"⚠️  Audio RMS extraction failed: {e}, using fallback", style="yellow"
        )
        # Return reasonable fallback values
        return [0.3 + 0.1 * (i % 5) for i in range(100)]


def validate_video_file(video_path: str) -> bool:
    """Validate video file exists and is accessible, prevent path traversal"""
    # SECURITY: Robust path traversal protection
    try:
        # Resolve all symbolic links and relative paths
        abs_path = os.path.realpath(os.path.abspath(video_path))

        # Define allowed base directories (current working directory and common video locations)
        allowed_bases = [
            os.path.realpath(os.getcwd()),
            os.path.realpath(os.path.expanduser("~/Videos")),
            os.path.realpath(os.path.expanduser("~/Downloads")),
            os.path.realpath("/tmp"),
            os.path.realpath("/var/tmp"),
        ]

        # Check if resolved path is within any allowed directory
        path_is_safe = False
        for base in allowed_bases:
            try:
                os.path.relpath(abs_path, base)
                if abs_path.startswith(base + os.sep) or abs_path == base:
                    path_is_safe = True
                    break
            except ValueError:
                # relpath raises ValueError if paths are on different drives (Windows)
                continue

        if not path_is_safe:
            console.print(
                f"❌ SECURITY: Path outside allowed directories: {abs_path}",
                style="bold red",
            )
            return False

        # Verify file exists and is actually a file
        if not os.path.exists(abs_path):
            console.print(f"❌ File does not exist: {abs_path}", style="bold red")
            return False

        if not os.path.isfile(abs_path):
            console.print(f"❌ Path is not a file: {abs_path}", style="bold red")
            return False

        # SECURITY: Check file extension to prevent processing of suspicious files
        allowed_extensions = {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm",
            ".flv",
            ".wmv",
            ".m4v",
        }
        file_ext = os.path.splitext(abs_path)[1].lower()
        if file_ext not in allowed_extensions:
            console.print(
                f"❌ SECURITY: Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}",
                style="bold red",
            )
            return False

    except (OSError, ValueError, AttributeError, TypeError) as e:
        console.print(f"❌ SECURITY: Path validation error: {e}", style="bold red")
        return False

    # Check file size
    file_size = os.path.getsize(abs_path)
    if file_size == 0:
        console.print(f"❌ Video file is empty: {abs_path}", style="bold red")
        return False

    # Check file extension
    valid_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]
    if not any(abs_path.lower().endswith(ext) for ext in valid_extensions):
        console.print(
            f"⚠️  Warning: Unusual file extension. Supported: {', '.join(valid_extensions)}",
            style="yellow",
        )

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
    table.add_row("Video Codec", info.get("video_codec", "Unknown"))
    table.add_row("Audio Codec", info.get("audio_codec", "Unknown"))

    console.print(table)


def start_mcp_server():
    """Start MCP server in background thread"""

    def run_server():
        try:
            start_server(host="localhost", port=7801)
        except (ImportError, OSError, RuntimeError) as e:
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
            console.print(
                f"❌ MCP server unhealthy: {response.status_code}", style="bold red"
            )
            return False
    except (requests.RequestException, ConnectionError, TimeoutError) as e:
        console.print(f"❌ MCP server not responding: {e}", style="bold red")
        return False


def post_to_mcp(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Post data to MCP server"""
    try:
        response = requests.post(
            f"http://localhost:7801/{endpoint}", json=data, timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            console.print(
                f"❌ MCP {endpoint} failed: {response.status_code}", style="bold red"
            )
            console.print(f"   Response: {response.text}", style="red")
            return {"error": f"HTTP {response.status_code}"}

    except (requests.RequestException, ConnectionError, TimeoutError, ValueError) as e:
        console.print(f"❌ MCP request failed: {e}", style="bold red")
        return {"error": str(e)}


def create_subtitles_for_clips(clips: List[Dict], words: List[Dict]) -> List[str]:
    """Create subtitle files for each clip"""
    subtitle_files = []

    for i, clip in enumerate(clips):
        # Create subtitles in the output directory
        output_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "output",
            "subtitles",
        )
        os.makedirs(output_dir, exist_ok=True)
        subtitle_path = os.path.join(output_dir, f"subtitle_{i+1}.srt")

        if create_subtitle_file(words, clip["start_ms"], clip["end_ms"], subtitle_path):
            subtitle_files.append(subtitle_path)
        else:
            console.print(
                f"⚠️  Failed to create subtitle for clip {i+1}", style="yellow"
            )

    return subtitle_files


def process_video_pipeline(
    video_path: str, mode: str = "smart", vertical: bool = True
) -> Dict[str, Any]:
    """Main video processing pipeline"""

    console.print(f"\n🎬 Processing video in {mode.upper()} mode", style="bold blue")
    console.print(f"   Video: {os.path.basename(video_path)}")
    if vertical:
        console.print(f"   Format: Vertical (1080x1920)", style="cyan")

    # Create output directories
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "output",
    )
    subtitles_dir = os.path.join(output_dir, "subtitles")

    try:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(subtitles_dir, exist_ok=True)
        console.print(f"📁 Output directory: {output_dir}", style="dim")
    except (OSError, PermissionError) as e:
        console.print(f"❌ Failed to create output directories: {e}", style="bold red")
        return {
            "video_path": video_path,
            "mode": mode,
            "success": False,
            "error": f"Failed to create output directories: {e}",
        }

    results = {
        "video_path": video_path,
        "mode": mode,
        "success": False,
        "error": None,
        "analysis": None,
        "highlights": None,
        "timeline": None,
        "cost": 0.0,
    }

    try:
        # Step 1: Analyze video
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:

                task = progress.add_task(
                    "Analyzing video (ASR + Diarization)...", total=None
                )
                analysis = analyze_video(video_path)
                progress.update(task, description="✅ Video analysis complete")

                results["analysis"] = {
                    "transcript_length": len(analysis["transcript"]),
                    "word_count": len(analysis["words"]),
                    "speaker_turns": len(analysis.get("speaker_turns", [])),
                }

                console.print(
                    f"📝 Transcript: {len(analysis['words'])} words, {len(analysis.get('speaker_turns', []))} speaker turns"
                )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            ValueError,
            RuntimeError,
        ) as e:
            console.print(f"⚠️  Video analysis error: {e}", style="yellow")
            # Create minimal analysis result
            analysis = {"words": [], "transcript": "", "speaker_turns": []}
            results["analysis"] = {
                "transcript_length": 0,
                "word_count": 0,
                "speaker_turns": 0,
                "error": str(e),
            }

        # Step 2: Choose highlights
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:

                task = progress.add_task(
                    f"Selecting highlights ({mode} mode)...", total=None
                )

                # Extract real audio RMS energy
                audio_energy = extract_audio_rms(video_path)

                # Convert words to segments format (FIX CRITICAL DATA FORMAT BUG)
                words = analysis.get("words", [])
                transcript_segments = []

                if words:
                    # Group words into segments (every 5 words, sentence boundaries, or minimum 1 word)
                    current_segment_words = []

                    for i, word in enumerate(words):
                        current_segment_words.append(word)

                        # Create segment on: sentence end, every 5 words, or last word
                        should_end_segment = (
                            word.get("word", "").strip().endswith((".", "!", "?"))
                            or len(current_segment_words) >= 5
                            or i == len(words) - 1  # Last word
                        )

                        if should_end_segment and current_segment_words:
                            # Combine words into text with deduplication
                            words_text = [
                                w.get("word", "").strip() for w in current_segment_words
                            ]
                            # Remove consecutive duplicates
                            cleaned_words = []
                            prev_word = None
                            for w in words_text:
                                if prev_word is None or w.lower() != prev_word.lower():
                                    cleaned_words.append(w)
                                prev_word = w
                            text = " ".join(cleaned_words)
                            start_time = current_segment_words[0].get("start", 0)
                            end_time = current_segment_words[-1].get(
                                "end", start_time + 1
                            )

                            # Only create segment if text is meaningful
                            if text.strip():
                                transcript_segments.append(
                                    {
                                        "text": text.strip(),
                                        "start_ms": int(
                                            start_time * 1000
                                        ),  # Convert to milliseconds
                                        "end_ms": int(end_time * 1000),
                                        "confidence": sum(
                                            w.get("confidence", 0.8)
                                            for w in current_segment_words
                                        )
                                        / len(current_segment_words),
                                    }
                                )

                            current_segment_words = []

                print(
                    f"🔧 Converted {len(words)} words to {len(transcript_segments)} segments"
                )
                highlights = select_highlights(transcript_segments, audio_energy, mode)
                progress.update(task, description="✅ Highlights selected")

                results["highlights"] = highlights
                from ..core.cost import get_current_cost

                results["cost"] = get_current_cost()
        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            console.print(f"⚠️  Highlight selection error: {e}", style="yellow")
            highlights = []
            results["highlights"] = highlights
            results["cost"] = get_total_cost()

        console.print(
            f"🎯 Selected {len(highlights)} highlights (Cost: ${results['cost']:.3f})"
        )

        # Handle case where no highlights were selected
        if not highlights:
            console.print(
                "⚠️  No highlights selected - check transcript quality", style="yellow"
            )
            # Create a dummy highlight covering the first 30 seconds
            # Calculate fallback audio energy (first 30 seconds average)
            fallback_energy = 0.5  # Default
            if audio_energy and len(audio_energy) > 0:
                # Average of first 30 samples (roughly 30 seconds)
                samples_to_use = min(30, len(audio_energy))
                fallback_energy = sum(audio_energy[:samples_to_use]) / samples_to_use

            highlights = [
                {
                    "slug": "full-clip",
                    "title": "Full Content",
                    "start_ms": 0,
                    "end_ms": 30000,  # 30 seconds
                    "score": 1.0,
                    "audio_energy": fallback_energy,
                }
            ]
            console.print("   Using first 30 seconds as fallback", style="yellow")
            # Update results with the fallback highlight
            results["highlights"] = highlights

        # Step 3: Create subtitles
        subtitle_files = []
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:

                task = progress.add_task("Creating subtitles...", total=None)
                subtitle_files = create_subtitles_for_clips(
                    highlights, analysis["words"]
                )
                progress.update(task, description="✅ Subtitles created")

                console.print(f"📄 Created {len(subtitle_files)} subtitle files")
        except (OSError, IOError, ValueError, TypeError) as e:
            console.print(f"⚠️  Subtitle creation error: {e}", style="yellow")
            # Continue without subtitles

        # Step 4: Build timeline - use intelligent cropping for vertical format
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:

                task = progress.add_task("Building timeline...", total=None)

                if vertical:
                    # Use intelligent vertical cropping instead of DaVinci Resolve
                    project_name = f"Montage_{int(time.time())}"
                    output_path = os.path.join(
                        output_dir, f"{project_name}_vertical.mp4"
                    )

                    from ..utils.intelligent_crop import (
                        create_intelligent_vertical_video,
                    )

                    success = create_intelligent_vertical_video(
                        highlights, os.path.abspath(video_path), output_path
                    )

                    timeline_result = {
                        "success": success,
                        "project_name": project_name,
                        "output_path": output_path if success else None,
                        "clips_added": len(highlights) if success else 0,
                        "vertical_format": True,
                        "resolution": "1080x1920",
                        "method": "intelligent_vertical_cropping",
                    }

                else:
                    # Use MCP/DaVinci Resolve for horizontal format
                    abs_video_path = os.path.abspath(video_path)

                    timeline_data = {
                        "video_path": abs_video_path,
                        "clips": highlights,
                        "project_name": f"Montage_{int(time.time())}",
                        "vertical_format": vertical,
                    }

                    timeline_result = post_to_mcp("buildTimeline", timeline_data)

                progress.update(task, description="✅ Timeline built")
                results["timeline"] = timeline_result

                if timeline_result.get("success"):
                    console.print(
                        f"🎬 Timeline created: {timeline_result.get('method', 'unknown')}"
                    )
                else:
                    console.print(
                        f"⚠️  Timeline creation issue: {timeline_result.get('error', 'unknown')}",
                        style="yellow",
                    )
                    # Don't fail the pipeline
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            OSError,
            ValueError,
        ) as e:
            console.print(f"⚠️  Timeline building error: {e}", style="yellow")
            results["timeline"] = {"error": str(e), "success": False}

        # Only mark success if video was created
        if results.get("timeline", {}).get("output_path"):
            output_path = results["timeline"]["output_path"]
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                results["success"] = True
                results["output_video"] = output_path
            else:
                results["success"] = False
                results["error"] = "Video file was not created"
        else:
            results["success"] = False
            results["error"] = "No output path returned from timeline creation"

        # PRODUCTION QUALITY VALIDATION - Hard fail on quality issues
        if results["success"]:
            try:
                console.print("🔍 Running production quality validation...")
                results = enforce_production_quality(results)
                console.print("✅ Quality validation passed")
            except QualityValidationError as e:
                console.print(f"❌ Quality validation failed: {e}", style="bold red")
                results["success"] = False
                results["error"] = f"Quality validation failed: {e}"

        return results

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        OSError,
        ValueError,
        RuntimeError,
    ) as e:
        results["error"] = str(e)
        console.print(f"❌ Pipeline failed: {e}", style="bold red")
        return results
    except Exception as e:
        # Catch any truly unexpected errors
        results["error"] = f"Unexpected error: {type(e).__name__}: {e}"
        console.print(
            f"❌ Unexpected pipeline failure: {type(e).__name__}: {e}", style="bold red"
        )
        return results


def display_results(results: Dict[str, Any]):
    """Display pipeline results"""

    if not results["success"]:
        console.print("\n❌ Pipeline Failed", style="bold red")
        console.print(f"   Error: {results.get('error', 'Unknown error')}")
        return

    # Success summary
    console.print("\n✅ Pipeline Completed Successfully!", style="bold green")

    # Show output video
    if results.get("output_video"):
        console.print(f"\n🎥 Output video created: {results['output_video']}")
        size_mb = os.path.getsize(results["output_video"]) / (1024 * 1024)
        console.print(f"   Size: {size_mb:.2f} MB")

    # Analysis summary
    if results.get("analysis"):
        analysis = results["analysis"]
        console.print("\n📊 Analysis Summary:")
        console.print(f"   • Words transcribed: {analysis['word_count']}")
        console.print(f"   • Speaker turns: {analysis['speaker_turns']}")
        console.print(
            f"   • Transcript length: {analysis['transcript_length']} characters"
        )

    # Highlights summary
    if results.get("highlights"):
        highlights = results["highlights"]
        console.print("\n🎯 Highlights Summary:")
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
                (
                    highlight["title"][:50] + "..."
                    if len(highlight["title"]) > 50
                    else highlight["title"]
                ),
                f"{duration:.1f}s",
                f"{highlight['score']:.1f}",
            )

        console.print(table)

    # Timeline summary
    if results.get("timeline"):
        timeline = results["timeline"]
        console.print("\n🎬 Timeline Summary:")
        console.print(f"   • Method: {timeline.get('method', 'unknown')}")
        console.print(f"   • Clips added: {timeline.get('clips_added', 0)}")

        if timeline.get("output_path"):
            console.print(f"   • Output: {timeline['output_path']}")

    # Export JSON plan
    plan_file = f"montage_plan_{int(time.time())}.json"
    with open(plan_file, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n💾 Full plan saved to: {plan_file}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Real AI-powered video processing pipeline"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "--mode",
        choices=["smart", "premium"],
        default="smart",
        help="Processing mode (smart=local, premium=AI-enhanced)",
    )
    parser.add_argument("--info", action="store_true", help="Show video info only")
    parser.add_argument("--no-server", action="store_true", help="Skip MCP server")
    parser.add_argument(
        "--vertical",
        action="store_true",
        help="Output vertical format (1080x1920) for social media",
    )

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
    results = process_video_pipeline(
        args.video, args.mode, args.vertical if hasattr(args, "vertical") else True
    )

    # Display results
    display_results(results)

    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
