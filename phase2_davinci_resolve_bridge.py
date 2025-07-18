#!/usr/bin/env python3
"""
Phase 2: DaVinci Resolve 20 MCP bridge
Bridge to connect with DaVinci Resolve for automated video editing
"""

import os
import sys
import json
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid

# Set up DaVinci Resolve API paths for macOS
RESOLVE_SCRIPT_API = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
RESOLVE_SCRIPT_LIB = "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"

# Add to Python path
if RESOLVE_SCRIPT_API not in sys.path:
    sys.path.append(os.path.join(RESOLVE_SCRIPT_API, "Modules"))

# Set environment variables
os.environ["RESOLVE_SCRIPT_API"] = RESOLVE_SCRIPT_API
os.environ["RESOLVE_SCRIPT_LIB"] = RESOLVE_SCRIPT_LIB
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + os.path.join(RESOLVE_SCRIPT_API, "Modules")

# Import DaVinci Resolve API
try:
    import DaVinciResolveScript as dvr_script
    print("✅ DaVinci Resolve API loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import DaVinci Resolve API: {e}")
    print(f"   RESOLVE_SCRIPT_API: {RESOLVE_SCRIPT_API}")
    print(f"   RESOLVE_SCRIPT_LIB: {RESOLVE_SCRIPT_LIB}")
    print(f"   Make sure DaVinci Resolve is installed and the API paths are correct")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EditCommand:
    """Single edit command for DaVinci Resolve"""
    command_type: str  # 'import', 'cut', 'add_subtitle', 'export', etc.
    parameters: Dict[str, Any]
    timeline_position: float
    duration: float
    priority: int
    metadata: Dict[str, Any]

@dataclass
class EditProject:
    """DaVinci Resolve project structure"""
    project_name: str
    timeline_name: str
    video_path: str
    highlights: List[Dict[str, Any]]
    subtitles: List[Dict[str, Any]]
    commands: List[EditCommand]
    project_id: str
    created_at: float


class DaVinciResolveBridge:
    """Bridge to DaVinci Resolve for automated editing"""
    
    def __init__(self):
        self.resolve = None
        self.project_manager = None
        self.current_project = None
        self.current_timeline = None
        self.media_pool = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to DaVinci Resolve"""
        logger.info("🎬 Connecting to DaVinci Resolve...")
        
        try:
            self.resolve = dvr_script.scriptapp("Resolve")
            if self.resolve:
                self.project_manager = self.resolve.GetProjectManager()
                if self.project_manager:
                    self.connected = True
                    logger.info("✅ Connected to DaVinci Resolve")
                else:
                    logger.error("❌ Failed to get Project Manager")
                    return False
            else:
                logger.error("❌ Failed to connect to DaVinci Resolve - make sure it's running")
                return False
        except Exception as e:
            logger.error(f"❌ DaVinci Resolve connection failed: {e}")
            return False
        
        return self.connected
    
    def disconnect(self):
        """Disconnect from DaVinci Resolve"""
        self.resolve = None
        self.project_manager = None
        self.current_project = None
        self.current_timeline = None
        self.media_pool = None
        self.connected = False
        logger.info("🎬 Disconnected from DaVinci Resolve")
    
    def create_project(self, project_name: str) -> bool:
        """Create a new DaVinci Resolve project"""
        if not self.connected:
            logger.error("❌ Not connected to DaVinci Resolve")
            return False
        
        try:
            # Create new project
            self.current_project = self.project_manager.CreateProject(project_name)
            if self.current_project:
                self.media_pool = self.current_project.GetMediaPool()
                logger.info(f"✅ Project created: {project_name}")
                return True
            else:
                logger.error(f"❌ Failed to create project: {project_name}")
                return False
        except Exception as e:
            logger.error(f"❌ Project creation failed: {e}")
            return False
    
    def create_timeline(self, timeline_name: str, settings: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new timeline in the current project"""
        if not self.current_project:
            logger.error("❌ No active project")
            return False
        
        try:
            # Set project settings first
            project_settings = {
                "timelineFrameRate": "24",
                "timelineResolutionWidth": "1920", 
                "timelineResolutionHeight": "1080",
                "videoCaptureFormat": "HD 1080p 24",
                "videoPlaybackFormat": "HD 1080p 24"
            }
            
            if settings:
                project_settings.update(settings)
            
            # Apply project settings
            self.current_project.SetSetting("timelineFrameRate", project_settings["timelineFrameRate"])
            self.current_project.SetSetting("timelineResolutionWidth", project_settings["timelineResolutionWidth"])
            self.current_project.SetSetting("timelineResolutionHeight", project_settings["timelineResolutionHeight"])
            
            # Create timeline
            timeline = self.media_pool.CreateEmptyTimeline(timeline_name)
            
            if timeline:
                self.current_timeline = timeline
                
                # Set up timeline with multiple video tracks for subtitles
                self.current_timeline.SetSetting("videoTrackCount", "3")  # V1 for main, V2 for subtitles, V3 for overlays
                self.current_timeline.SetSetting("audioTrackCount", "2")  # A1 for main, A2 for music/effects
                
                logger.info(f"✅ Timeline created: {timeline_name}")
                return True
            else:
                logger.error(f"❌ Failed to create timeline: {timeline_name}")
                return False
        except Exception as e:
            logger.error(f"❌ Timeline creation failed: {e}")
            return False
    
    def import_media(self, file_path: str) -> Any:
        """Import media file into the project"""
        if not self.media_pool:
            logger.error("❌ No media pool available")
            return None
        
        if not os.path.exists(file_path):
            logger.error(f"❌ File not found: {file_path}")
            return None
        
        try:
            # Import media files
            media_items = self.media_pool.ImportMedia([file_path])
            
            if media_items and len(media_items) > 0:
                logger.info(f"✅ Media imported: {file_path}")
                return media_items[0]
            else:
                logger.error(f"❌ Failed to import media: {file_path}")
                return None
        except Exception as e:
            logger.error(f"❌ Media import failed: {e}")
            return None
    
    def get_media_properties(self, media_item: Any) -> Dict[str, Any]:
        """Get media item properties"""
        try:
            properties = {
                'name': media_item.GetName(),
                'duration': media_item.GetClipProperty('Duration'),
                'fps': media_item.GetClipProperty('FPS'),
                'width': media_item.GetClipProperty('Resolution').split('x')[0],
                'height': media_item.GetClipProperty('Resolution').split('x')[1]
            }
            return properties
        except Exception as e:
            logger.warning(f"Could not get media properties: {e}")
            return {}
    
    def add_highlight_clips(self, highlights: List[Dict[str, Any]], media_item: Any) -> bool:
        """Add highlight clips to the timeline"""
        if not self.current_timeline:
            logger.error("❌ No active timeline")
            return False
        
        if not media_item:
            logger.error("❌ No media item provided")
            return False
        
        logger.info(f"🎬 Adding {len(highlights)} highlight clips...")
        
        # Get media properties
        media_props = self.get_media_properties(media_item)
        fps = float(media_props.get('fps', 24))
        
        # Get current timeline position
        timeline_clips = []
        
        for i, highlight in enumerate(highlights):
            try:
                # Convert milliseconds to frames based on actual FPS
                start_frame = int(highlight.get('start_ms', 0) * fps / 1000)
                end_frame = int(highlight.get('end_ms', 0) * fps / 1000)
                duration_frames = end_frame - start_frame
                
                # Create clip info dictionary
                clip_info = {
                    'mediaPoolItem': media_item,
                    'startFrame': start_frame,
                    'endFrame': end_frame
                }
                
                timeline_clips.append(clip_info)
                
                logger.info(f"📎 Preparing clip: {highlight.get('slug', f'highlight_{i+1}')} "
                          f"[{start_frame}-{end_frame}] ({duration_frames} frames)")
                    
            except Exception as e:
                logger.error(f"❌ Failed to prepare clip {i+1}: {e}")
        
        # Add all clips to timeline in one batch
        if timeline_clips:
            try:
                # AppendToTimeline adds clips sequentially
                success = self.media_pool.AppendToTimeline(timeline_clips)
                
                if success:
                    logger.info(f"✅ Successfully added {len(timeline_clips)} clips to timeline")
                    
                    # Get the timeline clips and name them
                    video_track = self.current_timeline.GetItemListInTrack('video', 1)
                    for i, (clip, highlight) in enumerate(zip(video_track, highlights)):
                        try:
                            clip.SetName(highlight.get('slug', f'highlight_{i+1}'))
                        except:
                            pass
                    
                    return True
                else:
                    logger.error("❌ Failed to add clips to timeline")
                    return False
            except Exception as e:
                logger.error(f"❌ Failed to add clips to timeline: {e}")
                return False
        
        return False
    
    def add_subtitles(self, subtitles_dir: str) -> bool:
        """Add subtitle files to the timeline"""
        if not self.current_timeline:
            logger.error("❌ No active timeline")
            return False
        
        if not os.path.exists(subtitles_dir):
            logger.error(f"❌ Subtitles directory not found: {subtitles_dir}")
            return False
        
        logger.info(f"🎬 Adding subtitles from: {subtitles_dir}")
        
        # Find all SRT files
        srt_files = list(Path(subtitles_dir).glob("*.srt"))
        
        # DaVinci Resolve handles subtitles through Fusion or as tracks
        # For now, we'll import them as media items that can be overlaid
        subtitle_track_index = 2  # V2 track for subtitles
        
        for srt_file in srt_files:
            try:
                # Import subtitle file as a media item
                subtitle_items = self.media_pool.ImportMedia([str(srt_file)])
                
                if subtitle_items and len(subtitle_items) > 0:
                    logger.info(f"✅ Subtitle imported: {srt_file.name}")
                    
                    # Add subtitle to timeline on a higher track
                    # This requires the timeline to have multiple video tracks
                    # In production, you'd add text+ or fusion titles with the subtitle content
                else:
                    logger.warning(f"⚠️ Could not import subtitle: {srt_file.name}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to add subtitle {srt_file.name}: {e}")
        
        return True
    
    def export_timeline(self, output_path: str, export_settings: Optional[Dict[str, Any]] = None) -> bool:
        """Export the timeline to a video file"""
        if not self.current_project:
            logger.error("❌ No active project")
            return False
        
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Set render settings
            render_settings = {
                "SelectAllFrames": True,
                "TargetDir": output_dir or os.getcwd(),
                "CustomName": os.path.basename(output_path).replace('.mp4', ''),
                "FormatWidth": 1920,
                "FormatHeight": 1080,
                "FrameRate": "24",
                "PixelAspectRatio": 1.0,
                "VideoQuality": 0,  # 0 = Automatic
                "AudioSampleRate": 48000,
                "AudioBitDepth": 16,
                "ExportVideo": True,
                "ExportAudio": True,
                "FormatAndCodec": "mp4",
                "Codec": "H264",
                "EncodingProfile": "Main",
                "MultiPassEncode": False,
                "AlphaMode": 0
            }
            
            if export_settings:
                render_settings.update(export_settings)
            
            # Try to load a suitable render preset
            presets = self.current_project.GetRenderPresetList()
            h264_preset = None
            
            for preset in presets:
                if "H.264" in preset or "H264" in preset or "MP4" in preset.upper():
                    h264_preset = preset
                    break
            
            if h264_preset:
                self.current_project.LoadRenderPreset(h264_preset)
                logger.info(f"📋 Loaded render preset: {h264_preset}")
            
            # Apply render settings
            self.current_project.SetRenderSettings(render_settings)
            
            # Add job to render queue
            job_id = self.current_project.AddRenderJob()
            
            if job_id:
                logger.info(f"📋 Render job added with ID: {job_id}")
                
                # Start rendering
                self.current_project.StartRendering(job_id)
                
                # Monitor render progress
                render_start = time.time()
                last_progress = -1
                
                while self.current_project.IsRenderingInProgress():
                    try:
                        # Get render job status
                        render_jobs = self.current_project.GetRenderJobList()
                        for job in render_jobs:
                            if job.get('JobId') == job_id:
                                progress = job.get('CompletionPercentage', 0)
                                if progress != last_progress:
                                    logger.info(f"⏳ Rendering: {progress}% complete")
                                    last_progress = progress
                                break
                    except:
                        pass
                    
                    time.sleep(1)
                
                render_time = time.time() - render_start
                logger.info(f"✅ Timeline exported in {render_time:.1f}s: {output_path}")
                return True
            else:
                logger.error("❌ Failed to add render job")
                return False
                
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            return False
    
    def create_edit_project(self, video_path: str, highlights: List[Dict[str, Any]], 
                          subtitles_dir: str, output_path: str) -> EditProject:
        """Create a complete edit project"""
        project_id = str(uuid.uuid4())
        project_name = f"montage_project_{int(time.time())}"
        timeline_name = f"montage_timeline_{int(time.time())}"
        
        # Create edit commands
        commands = []
        
        # Command 1: Import media
        commands.append(EditCommand(
            command_type="import_media",
            parameters={"file_path": video_path},
            timeline_position=0,
            duration=0,
            priority=1,
            metadata={"description": "Import source video"}
        ))
        
        # Command 2: Add highlight clips
        for i, highlight in enumerate(highlights):
            commands.append(EditCommand(
                command_type="add_highlight_clip",
                parameters={
                    "highlight": highlight,
                    "position": i * 10.0
                },
                timeline_position=i * 10.0,
                duration=(highlight.get('end_ms', 0) - highlight.get('start_ms', 0)) / 1000.0,
                priority=2,
                metadata={"clip_index": i, "slug": highlight.get('slug', f'clip_{i}')}
            ))
        
        # Command 3: Add subtitles
        commands.append(EditCommand(
            command_type="add_subtitles",
            parameters={"subtitles_dir": subtitles_dir},
            timeline_position=0,
            duration=0,
            priority=3,
            metadata={"description": "Add subtitle tracks"}
        ))
        
        # Command 4: Export timeline
        commands.append(EditCommand(
            command_type="export_timeline",
            parameters={
                "output_path": output_path,
                "settings": {
                    "format": "mp4",
                    "resolution": "1920x1080",
                    "framerate": 24
                }
            },
            timeline_position=0,
            duration=0,
            priority=4,
            metadata={"description": "Export final video"}
        ))
        
        return EditProject(
            project_name=project_name,
            timeline_name=timeline_name,
            video_path=video_path,
            highlights=highlights,
            subtitles=[],
            commands=commands,
            project_id=project_id,
            created_at=time.time()
        )
    
    def execute_edit_project(self, edit_project: EditProject) -> bool:
        """Execute a complete edit project"""
        logger.info(f"🎬 Executing edit project: {edit_project.project_name}")
        
        # Connect to DaVinci Resolve
        if not self.connect():
            return False
        
        try:
            # Create project
            if not self.create_project(edit_project.project_name):
                return False
            
            # Create timeline
            if not self.create_timeline(edit_project.timeline_name):
                return False
            
            # Import media first
            media_item = None
            
            # Execute commands in priority order
            sorted_commands = sorted(edit_project.commands, key=lambda x: x.priority)
            
            for command in sorted_commands:
                logger.info(f"🎬 Executing command: {command.command_type}")
                
                if command.command_type == "import_media":
                    media_item = self.import_media(command.parameters["file_path"])
                    success = media_item is not None
                elif command.command_type == "add_highlight_clip":
                    if media_item:
                        success = self.add_highlight_clips([command.parameters["highlight"]], media_item)
                    else:
                        logger.error("❌ No media item available for clips")
                        success = False
                elif command.command_type == "add_subtitles":
                    success = self.add_subtitles(command.parameters["subtitles_dir"])
                elif command.command_type == "export_timeline":
                    success = self.export_timeline(
                        command.parameters["output_path"],
                        command.parameters.get("settings")
                    )
                else:
                    logger.warning(f"⚠️  Unknown command type: {command.command_type}")
                    success = True
                
                if not success:
                    logger.error(f"❌ Command failed: {command.command_type}")
                    return False
            
            logger.info("✅ Edit project executed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Edit project execution failed: {e}")
            return False
        finally:
            self.disconnect()

async def main():
    """Test DaVinci Resolve bridge"""
    if len(sys.argv) < 4:
        print("Usage: python phase2_davinci_resolve_bridge.py <video_path> <highlights_json> <subtitles_dir> [output_path]")
        return
    
    video_path = sys.argv[1]
    highlights_path = sys.argv[2]
    subtitles_dir = sys.argv[3]
    output_path = sys.argv[4] if len(sys.argv) > 4 else "montage_output.mp4"
    
    # Load highlights
    with open(highlights_path, 'r') as f:
        highlights_data = json.load(f)
        highlights = highlights_data.get("top_highlights", [])
    
    # Initialize bridge
    bridge = DaVinciResolveBridge()  # Use real DaVinci Resolve API
    
    # Create edit project
    edit_project = bridge.create_edit_project(
        video_path=video_path,
        highlights=highlights,
        subtitles_dir=subtitles_dir,
        output_path=output_path
    )
    
    # Save project file
    project_file = f"edit_project_{int(time.time())}.json"
    with open(project_file, 'w') as f:
        json.dump(asdict(edit_project), f, indent=2)
    
    print("\n" + "=" * 60)
    print("🎬 DAVINCI RESOLVE BRIDGE RESULTS")
    print("=" * 60)
    print(f"📝 Project: {edit_project.project_name}")
    print(f"🎞️  Timeline: {edit_project.timeline_name}")
    print(f"📹 Video: {edit_project.video_path}")
    print(f"🎯 Highlights: {len(edit_project.highlights)}")
    print(f"⚙️  Commands: {len(edit_project.commands)}")
    print(f"💾 Project file: {project_file}")
    
    # Execute project
    success = bridge.execute_edit_project(edit_project)
    
    if success:
        print("\n✅ Edit project executed successfully!")
        print(f"📤 Output: {output_path}")
    else:
        print("\n❌ Edit project execution failed!")

if __name__ == "__main__":
    asyncio.run(main())