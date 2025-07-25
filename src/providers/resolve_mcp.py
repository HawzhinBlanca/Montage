#!/usr/bin/env python3
"""
DaVinci Resolve MCP bridge - Bottle server exposing /buildTimeline + /renderProxy
Uses DaVinciResolveScript when available; falls back to noop
"""
import os
import sys
from typing import Dict, List
from bottle import Bottle, request, response, run

# Try to import DaVinci Resolve API
RESOLVE_AVAILABLE = False
try:
    # Set up DaVinci Resolve API paths
    RESOLVE_SCRIPT_API = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
    RESOLVE_SCRIPT_LIB = "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"

    if RESOLVE_SCRIPT_API not in sys.path:
        sys.path.append(os.path.join(RESOLVE_SCRIPT_API, "Modules"))

    os.environ["RESOLVE_SCRIPT_API"] = RESOLVE_SCRIPT_API
    os.environ["RESOLVE_SCRIPT_LIB"] = RESOLVE_SCRIPT_LIB

    import DaVinciResolveScript as dvr_script

    RESOLVE_AVAILABLE = True
    print("✅ DaVinci Resolve API available")

except ImportError:
    print("⚠️  DaVinci Resolve API not available - using fallback mode")
    RESOLVE_AVAILABLE = False

app = Bottle()


class ResolveManager:
    """Manager for DaVinci Resolve operations"""

    def __init__(self):
        self.resolve = None
        self.project_manager = None
        self.current_project = None
        self.media_pool = None
        self.connected = False

        if RESOLVE_AVAILABLE:
            self.connect()

    def connect(self) -> bool:
        """Connect to DaVinci Resolve"""
        if not RESOLVE_AVAILABLE:
            return False

        try:
            self.resolve = dvr_script.scriptapp("Resolve")
            if self.resolve:
                self.project_manager = self.resolve.GetProjectManager()
                self.connected = True
                print("✅ Connected to DaVinci Resolve")
                return True
            else:
                print("❌ DaVinci Resolve not running")
                return False
        except (AttributeError, RuntimeError, OSError) as e:
            print(f"❌ Failed to connect to DaVinci Resolve: {e}")
            return False

    def create_project(self, name: str) -> bool:
        """Create new project"""
        if not self.connected:
            return False

        try:
            self.current_project = self.project_manager.CreateProject(name)
            if self.current_project:
                self.media_pool = self.current_project.GetMediaPool()
                return True
            return False
        except (AttributeError, RuntimeError, TypeError) as e:
            print(f"❌ Failed to create project: {e}")
            return False

    def import_media(self, file_path: str):
        """Import media file"""
        if not self.connected or not self.media_pool:
            return None

        try:
            media_items = self.media_pool.ImportMedia([file_path])
            return media_items[0] if media_items else None
        except (AttributeError, RuntimeError, FileNotFoundError, OSError) as e:
            print(f"❌ Failed to import media: {e}")
            return None

    def create_timeline(self, name: str):
        """Create new timeline"""
        if not self.connected or not self.media_pool:
            return None

        try:
            timeline = self.media_pool.CreateEmptyTimeline(name)
            return timeline
        except (AttributeError, RuntimeError, ValueError) as e:
            print(f"❌ Failed to create timeline: {e}")
            return None

    def add_clips_to_timeline(self, timeline, media_item, clips: List[Dict]) -> bool:
        """Add clips to timeline"""
        if not timeline or not media_item:
            return False

        try:
            # Convert clips to timeline items
            timeline_items = []
            for clip in clips:
                start_frame = int(
                    clip["start_ms"] * 24 / 1000
                )  # Convert to frames (24fps)
                end_frame = int(clip["end_ms"] * 24 / 1000)

                timeline_items.append(
                    {
                        "mediaPoolItem": media_item,
                        "startFrame": start_frame,
                        "endFrame": end_frame,
                    }
                )

            # Add to timeline
            success = self.media_pool.AppendToTimeline(timeline_items)
            return success

        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            print(f"❌ Failed to add clips to timeline: {e}")
            return False

    def render_proxy(self, timeline, output_path: str) -> bool:
        """Render proxy/preview"""
        if not timeline or not self.current_project:
            return False

        try:
            # Set render settings for proxy
            render_settings = {
                "SelectAllFrames": True,
                "TargetDir": os.path.dirname(output_path),
                "CustomName": os.path.basename(output_path).replace(".mp4", ""),
                "FormatWidth": 960,
                "FormatHeight": 540,
                "FrameRate": "24",
                "VideoQuality": 2,  # Lower quality for proxy
                "ExportVideoFormat": "MP4",
                "VideoFormat": "MP4",
                "FormatAndCodec": "MP4",
                "Codec": "H264",
            }

            self.current_project.SetRenderSettings(render_settings)

            # Add render job
            job_id = self.current_project.AddRenderJob()
            if job_id:
                # Start rendering
                self.current_project.StartRendering(job_id)

                # Wait for completion (simplified)
                while self.current_project.IsRenderingInProgress():
                    import time

                    time.sleep(1)

                return True

            return False

        except (AttributeError, RuntimeError, OSError, ValueError) as e:
            print(f"❌ Proxy render failed: {e}")
            return False

    def render_final(self, timeline, output_path: str, vertical: bool = False) -> bool:
        """Render final high-quality video"""
        if not timeline or not self.current_project:
            return False

        try:
            # Set render settings for final output
            if vertical:
                # Vertical format for social media
                render_settings = {
                    "SelectAllFrames": True,
                    "TargetDir": os.path.dirname(output_path),
                    "CustomName": os.path.basename(output_path).replace(".mp4", ""),
                    "FormatWidth": 1080,
                    "FormatHeight": 1920,
                    "FrameRate": "30",
                    "VideoQuality": 0,  # Highest quality
                    "ExportVideoFormat": "MP4",
                    "VideoFormat": "MP4",
                    "FormatAndCodec": "MP4",
                    "Codec": "H264",
                    "AudioCodec": "AAC",
                    "AudioBitrate": 192000,
                    "VideoBitrate": 10000000,  # 10 Mbps for high quality
                }
            else:
                # Standard landscape format
                render_settings = {
                    "SelectAllFrames": True,
                    "TargetDir": os.path.dirname(output_path),
                    "CustomName": os.path.basename(output_path).replace(".mp4", ""),
                    "FormatWidth": 1920,
                    "FormatHeight": 1080,
                    "FrameRate": "30",
                    "VideoQuality": 0,  # Highest quality
                    "ExportVideoFormat": "MP4",
                    "VideoFormat": "MP4",
                    "FormatAndCodec": "MP4",
                    "Codec": "H264",
                    "AudioCodec": "AAC",
                    "AudioBitrate": 192000,
                    "VideoBitrate": 15000000,  # 15 Mbps for high quality
                }

            self.current_project.SetRenderSettings(render_settings)

            # Add render job
            job_id = self.current_project.AddRenderJob()
            if job_id:
                print(f"🎬 Starting DaVinci Resolve render: {output_path}")

                # Start rendering
                self.current_project.StartRendering(job_id)

                # Wait for completion with progress
                import time

                while self.current_project.IsRenderingInProgress():
                    time.sleep(2)
                    print(".", end="", flush=True)

                print("\n✅ DaVinci Resolve render complete")

                # Find the actual file that was created and move it to expected path
                output_dir = os.path.dirname(output_path)
                timeline_files = []
                for f in os.listdir(output_dir):
                    if "timeline" in f and (f.endswith(".mov") or f.endswith(".mp4")):
                        file_path = os.path.join(output_dir, f)
                        timeline_files.append((file_path, os.path.getctime(file_path)))

                if timeline_files:
                    # Get the most recently created timeline file
                    actual_file = max(timeline_files, key=lambda x: x[1])[0]

                    # Move it to the expected path
                    if actual_file != output_path:
                        import shutil

                        shutil.move(actual_file, output_path)
                        print(
                            f"✅ Moved {os.path.basename(actual_file)} to {os.path.basename(output_path)}"
                        )

                return True

            return False

        except (
            AttributeError,
            RuntimeError,
            OSError,
            ValueError,
            FileNotFoundError,
        ) as e:
            print(f"❌ Final render failed: {e}")
            return False


# Global resolve manager
resolve_manager = ResolveManager()


@app.route("/buildTimeline", method="POST")
def build_timeline():
    """Build timeline from clip data with vertical 1080x1920 format"""
    try:
        data = request.json

        if not data:
            response.status = 400
            return {"error": "No JSON data provided"}

        video_path = data.get("video_path")
        clips = data.get("clips", [])
        project_name = data.get("project_name", "MontageProject")
        vertical_format = data.get("vertical_format", True)  # Default to vertical

        if not video_path or not clips:
            response.status = 422
            return {"error": "Missing required fields: video_path and clips"}

        # Try DaVinci Resolve first if available
        use_ffmpeg = True
        timeline = None

        if RESOLVE_AVAILABLE and resolve_manager.connected:
            # Real DaVinci Resolve implementation with render

            # Create project
            if not resolve_manager.create_project(project_name):
                use_ffmpeg = True
            else:
                # Import media
                media_item = resolve_manager.import_media(video_path)
                if not media_item:
                    use_ffmpeg = True
                else:
                    # Create timeline
                    timeline = resolve_manager.create_timeline("Main Timeline")
                    if not timeline:
                        use_ffmpeg = True
                    else:
                        # Add clips
                        if not resolve_manager.add_clips_to_timeline(
                            timeline, media_item, clips
                        ):
                            use_ffmpeg = True
                        else:
                            print("✅ DaVinci timeline created")

                            # Try to render with DaVinci Resolve
                            import os

                            output_dir = os.path.join(
                                os.path.dirname(
                                    os.path.dirname(
                                        os.path.dirname(os.path.abspath(__file__))
                                    )
                                ),
                                "output",
                            )
                            os.makedirs(output_dir, exist_ok=True)
                            output_path = os.path.join(
                                output_dir, f"{project_name}_timeline.mp4"
                            )

                            if resolve_manager.render_final(
                                timeline, output_path, vertical_format
                            ):
                                # Successfully rendered with DaVinci
                                use_ffmpeg = False
                                return {
                                    "success": True,
                                    "project_name": project_name,
                                    "output_path": output_path,
                                    "clips_added": len(clips),
                                    "vertical_format": vertical_format,
                                    "resolution": (
                                        "1080x1920" if vertical_format else "1920x1080"
                                    ),
                                    "method": "davinci_resolve_rendered",
                                }
                            else:
                                print(
                                    "⚠️  DaVinci render failed, falling back to FFmpeg"
                                )
                                use_ffmpeg = True

        if use_ffmpeg:
            # Fallback: Use FFmpeg for concatenation with vertical format
            from ..utils.ffmpeg_utils import concatenate_video_segments

            # Create output path in project output directory
            import os

            output_dir = os.path.join(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ),
                "output",
            )
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{project_name}_timeline.mp4")

            # Concatenate clips with vertical format option
            if concatenate_video_segments(
                clips, video_path, output_path, vertical_format=vertical_format
            ):
                return {
                    "success": True,
                    "project_name": project_name,
                    "output_path": output_path,
                    "clips_added": len(clips),
                    "vertical_format": vertical_format,
                    "resolution": "1080x1920" if vertical_format else "source",
                    "method": "ffmpeg_fallback",
                }
            else:
                response.status = 500
                return {"error": "Failed to concatenate clips"}

    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OSError,
        FileNotFoundError,
    ) as e:
        response.status = 500
        return {"error": str(e)}


@app.route("/renderProxy", method="POST")
def render_proxy():
    """Render proxy/preview video"""
    try:
        data = request.json

        if not data:
            response.status = 400
            return {"error": "No JSON data provided"}

        output_path = data.get("output_path", "/tmp/proxy_render.mp4")

        if (
            RESOLVE_AVAILABLE
            and resolve_manager.connected
            and resolve_manager.current_project
        ):
            # Real DaVinci Resolve proxy render

            # Get current timeline
            timeline = resolve_manager.current_project.GetCurrentTimeline()
            if not timeline:
                response.status = 500
                return {"error": "No active timeline"}

            # Render proxy
            if resolve_manager.render_proxy(timeline, output_path):
                return {
                    "success": True,
                    "output_path": output_path,
                    "method": "davinci_resolve",
                }
            else:
                response.status = 500
                return {"error": "Proxy render failed"}

        else:
            # Fallback: Copy/convert existing output
            try:
                import shutil

                # Find the most recent timeline output
                timeline_files = []
                for f in os.listdir("/tmp"):
                    if f.endswith("_timeline.mp4"):
                        timeline_files.append(f"/tmp/{f}")

                if timeline_files:
                    # Copy most recent file as proxy
                    latest_file = max(timeline_files, key=os.path.getctime)
                    shutil.copy(latest_file, output_path)

                    return {
                        "success": True,
                        "output_path": output_path,
                        "method": "ffmpeg_fallback",
                    }
                else:
                    response.status = 500
                    return {"error": "No timeline to render"}

            except (OSError, FileNotFoundError, PermissionError, AttributeError) as e:
                response.status = 500
                return {"error": f"Fallback render failed: {e}"}

    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OSError,
        FileNotFoundError,
    ) as e:
        response.status = 500
        return {"error": str(e)}


@app.route("/status", method="GET")
def status():
    """Get MCP bridge status"""
    return {
        "resolve_available": RESOLVE_AVAILABLE,
        "resolve_connected": resolve_manager.connected if resolve_manager else False,
        "current_project": (
            resolve_manager.current_project is not None if resolve_manager else False
        ),
        "version": "1.0.0",
    }


@app.route("/health", method="GET")
def health():
    """Health check endpoint"""
    return {"status": "healthy"}


def start_server(host="localhost", port=7801):
    """Start the MCP bridge server"""
    print(f"🚀 Starting MCP bridge server on {host}:{port}")
    print(
        f"   DaVinci Resolve: {'Available' if RESOLVE_AVAILABLE else 'Not available'}"
    )
    print("   Endpoints: /buildTimeline, /renderProxy, /status, /health")

    run(app, host=host, port=port, debug=False)


if __name__ == "__main__":
    start_server()
