# MONTAGE: 100% COMPLETE IMPLEMENTATION PLAN
## Honest, Real Functionality Only - No Fake Components

---

## 🎯 EXECUTIVE SUMMARY

**Goal**: Transform Montage into a 100% working, production-ready video processing pipeline with no fake/mock components.

**Approach**: Remove all overhyped "AI" features, focus on solid video processing technology that actually works.

**Timeline**: 14 tasks over 3 phases (Foundation → Core Features → Production)

---

## 📋 PHASE 1: FOUNDATION CLEANUP (Tasks 1-5)

### Task 1: Remove Fake AI Components [HIGH PRIORITY]
**Objective**: Clean codebase of all fake/misleading functionality

**Actions**:
1. **Delete fake modules**:
   ```bash
   rm src/core/emotion_analyzer.py
   rm src/core/narrative_detector.py
   rm src/core/speaker_analysis.py (keep real parts only)
   rm src/utils/video_effects.py (mostly stubs)
   ```

2. **Audit remaining files**:
   - Remove fake imports: `from core.emotion_analyzer import *`
   - Remove fake function calls in highlight_selector.py
   - Remove overhyped docstrings and comments

3. **Update documentation**:
   - Remove all "AI-powered", "advanced emotion", "narrative intelligence" claims
   - Replace with honest descriptions: "keyword-based content analysis"

**Deliverable**: Clean codebase with only real functionality
**Time**: 2 hours

### Task 2: Clean Core Transcription Module [HIGH PRIORITY]
**Objective**: Create robust, honest transcription system

**Implementation**:
```python
# src/core/transcription.py
class RealTranscriptionService:
    def __init__(self):
        self.whisper_model = None
        self.deepgram_client = None
    
    def transcribe_with_whisper(self, audio_path: str) -> dict:
        """Real Whisper transcription with word timestamps"""
        # Load model on demand
        if not self.whisper_model:
            self.whisper_model = whisper.load_model("base")
        
        result = self.whisper_model.transcribe(
            audio_path, 
            language="en",
            word_timestamps=True
        )
        
        return self._format_transcript(result)
    
    def transcribe_with_deepgram(self, audio_path: str) -> dict:
        """Real Deepgram API transcription"""
        # Implementation with real API calls
        pass
    
    def _format_transcript(self, raw_result: dict) -> dict:
        """Convert to standardized format"""
        segments = []
        for segment in raw_result["segments"]:
            segments.append({
                "text": segment["text"].strip(),
                "start_ms": int(segment["start"] * 1000),
                "end_ms": int(segment["end"] * 1000),
                "confidence": segment.get("avg_logprob", 0.0),
                "words": segment.get("words", [])
            })
        
        return {
            "segments": segments,
            "language": raw_result.get("language", "en"),
            "duration": raw_result.get("segments", [{}])[-1].get("end", 0)
        }
```

**Features**:
- Real Whisper model loading and transcription
- Real Deepgram API integration with error handling
- Standardized output format
- Word-level timestamps for precise editing

**Deliverable**: Production-ready transcription module
**Time**: 4 hours

### Task 3: Robust Face Detection System [HIGH PRIORITY]
**Objective**: Implement reliable face detection for intelligent cropping

**Implementation**:
```python
# src/core/face_detection.py
class FaceDetectionService:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
    
    def analyze_video_segment(self, video_path: str, start_ms: int, end_ms: int) -> dict:
        """Analyze faces in video segment for optimal cropping"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        start_frame = int((start_ms / 1000) * fps)
        end_frame = int((end_ms / 1000) * fps)
        
        face_positions = []
        frames_analyzed = 0
        
        # Sample frames throughout segment
        frame_interval = max(1, (end_frame - start_frame) // 10)
        
        for frame_idx in range(start_frame, end_frame, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            faces = self._detect_faces_multi_method(frame)
            if faces:
                face_positions.extend(self._calculate_face_centers(faces, frame.shape))
            
            frames_analyzed += 1
        
        cap.release()
        
        return self._calculate_optimal_crop(face_positions, frames_analyzed)
    
    def _detect_faces_multi_method(self, frame) -> list:
        """Use multiple detection methods for reliability"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Frontal faces
        faces = list(self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        ))
        
        # Method 2: Profile faces
        profiles = list(self.profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        ))
        
        faces.extend(profiles)
        return self._remove_overlapping_faces(faces)
    
    def _calculate_optimal_crop(self, face_positions: list, frames_analyzed: int) -> dict:
        """Calculate optimal crop center based on face data"""
        if not face_positions:
            return {
                "crop_center": (0.5, 0.5),
                "confidence": 0.0,
                "faces_detected": 0,
                "method": "center_fallback"
            }
        
        # Weight recent faces more heavily
        avg_x = sum(pos[0] for pos in face_positions) / len(face_positions)
        avg_y = sum(pos[1] for pos in face_positions) / len(face_positions)
        
        # Apply rule of thirds adjustment
        optimal_y = max(0.25, min(0.75, avg_y * 0.9))
        optimal_x = max(0.25, min(0.75, avg_x))
        
        confidence = min(1.0, len(face_positions) / (frames_analyzed * 2))
        
        return {
            "crop_center": (optimal_x, optimal_y),
            "confidence": confidence,
            "faces_detected": len(face_positions),
            "method": "face_based"
        }
```

**Features**:
- Multi-method face detection (frontal + profile)
- Overlapping face removal
- Confidence scoring
- Rule of thirds positioning
- Fallback to center crop when no faces

**Deliverable**: Reliable face detection service
**Time**: 6 hours

### Task 4: Professional Video Processing Pipeline [HIGH PRIORITY]
**Objective**: Build robust FFmpeg-based video processing

**Implementation**:
```python
# src/core/video_processor.py
class VideoProcessor:
    def __init__(self):
        self.face_detector = FaceDetectionService()
        self.temp_dir = "/tmp/montage_processing"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def process_video_segments(self, segments: list, input_video: str, output_path: str) -> bool:
        """Process video segments with intelligent cropping and encoding"""
        
        # Validate input
        if not self._validate_video(input_video):
            raise ValueError(f"Invalid input video: {input_video}")
        
        # Get video metadata
        metadata = self._get_video_metadata(input_video)
        
        processed_clips = []
        
        try:
            for i, segment in enumerate(segments):
                logger.info(f"Processing segment {i+1}/{len(segments)}")
                
                # Analyze faces for this segment
                face_analysis = self.face_detector.analyze_video_segment(
                    input_video, segment['start_ms'], segment['end_ms']
                )
                
                # Generate crop parameters
                crop_params = self._calculate_crop_parameters(
                    metadata, face_analysis['crop_center']
                )
                
                # Process segment
                clip_path = self._process_single_segment(
                    input_video, segment, crop_params, i
                )
                
                if clip_path:
                    processed_clips.append(clip_path)
                    logger.info(f"✅ Segment {i+1} processed: {face_analysis['faces_detected']} faces detected")
                else:
                    logger.error(f"❌ Failed to process segment {i+1}")
            
            # Concatenate all clips
            if processed_clips:
                success = self._concatenate_clips(processed_clips, output_path)
                self._cleanup_temp_files(processed_clips)
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            self._cleanup_temp_files(processed_clips)
            return False
    
    def _process_single_segment(self, input_video: str, segment: dict, crop_params: dict, index: int) -> str:
        """Process single video segment with cropping and encoding"""
        
        output_path = f"{self.temp_dir}/segment_{index:03d}.mp4"
        
        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y", "-v", "warning",
            "-ss", str(segment['start_ms'] / 1000),
            "-t", str((segment['end_ms'] - segment['start_ms']) / 1000),
            "-i", input_video,
            "-vf", self._build_video_filter(crop_params),
            "-c:v", "libx264", 
            "-preset", "medium", 
            "-crf", "18",
            "-c:a", "aac", 
            "-b:a", "128k",
            "-movflags", "+faststart",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return output_path
        else:
            logger.error(f"FFmpeg error: {result.stderr}")
            return None
    
    def _build_video_filter(self, crop_params: dict) -> str:
        """Build FFmpeg video filter string"""
        filters = []
        
        # Crop filter
        if crop_params['needs_crop']:
            filters.append(
                f"crop={crop_params['crop_width']}:"
                f"{crop_params['crop_height']}:"
                f"{crop_params['crop_x']}:"
                f"{crop_params['crop_y']}"
            )
        
        # Scale to target resolution
        filters.append("scale=1080:1920")
        
        # Set framerate
        filters.append("fps=30")
        
        return ",".join(filters)
    
    def _concatenate_clips(self, clip_paths: list, output_path: str) -> bool:
        """Concatenate processed clips into final video"""
        
        concat_file = f"{self.temp_dir}/concat_list.txt"
        
        with open(concat_file, "w") as f:
            for clip_path in clip_paths:
                f.write(f"file '{clip_path}'\n")
        
        cmd = [
            "ffmpeg", "-y", "-v", "warning",
            "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c", "copy",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        try:
            os.unlink(concat_file)
        except:
            pass
        
        return result.returncode == 0
```

**Features**:
- Professional FFmpeg integration
- Intelligent cropping based on face detection
- Error handling and recovery
- Temporary file management
- Progress logging

**Deliverable**: Production-ready video processor
**Time**: 8 hours

### Task 5: Real Segment Selection [HIGH PRIORITY]
**Objective**: Implement practical segment selection based on real criteria

**Implementation**:
```python
# src/core/segment_selector.py
class SegmentSelector:
    def __init__(self):
        self.min_duration_ms = 5000   # 5 seconds minimum
        self.max_duration_ms = 30000  # 30 seconds maximum
        self.min_confidence = -1.0    # Whisper confidence threshold
    
    def select_best_segments(self, transcript: dict, max_segments: int = 5) -> list:
        """Select best segments based on real quality metrics"""
        
        segments = transcript.get('segments', [])
        if not segments:
            return []
        
        # Filter by quality criteria
        quality_segments = self._filter_by_quality(segments)
        
        # Score segments
        scored_segments = self._score_segments(quality_segments)
        
        # Remove overlapping segments
        non_overlapping = self._remove_overlaps(scored_segments)
        
        # Select top segments
        selected = sorted(non_overlapping, key=lambda x: x['score'], reverse=True)[:max_segments]
        
        # Sort by timeline order
        selected.sort(key=lambda x: x['start_ms'])
        
        logger.info(f"Selected {len(selected)} segments from {len(segments)} total")
        return selected
    
    def _filter_by_quality(self, segments: list) -> list:
        """Filter segments by quality criteria"""
        quality_segments = []
        
        for segment in segments:
            duration = segment['end_ms'] - segment['start_ms']
            text_length = len(segment['text'].strip())
            confidence = segment.get('confidence', 0)
            
            # Quality checks
            if (self.min_duration_ms <= duration <= self.max_duration_ms and
                text_length >= 20 and  # At least 20 characters
                confidence >= self.min_confidence and
                self._has_complete_sentence(segment['text'])):
                
                quality_segments.append(segment)
        
        return quality_segments
    
    def _score_segments(self, segments: list) -> list:
        """Score segments based on multiple factors"""
        scored_segments = []
        
        for segment in segments:
            duration_s = (segment['end_ms'] - segment['start_ms']) / 1000
            text_length = len(segment['text'].strip())
            confidence = segment.get('confidence', 0)
            
            # Scoring factors
            duration_score = min(1.0, duration_s / 20)  # Optimal around 20 seconds
            length_score = min(1.0, text_length / 100)  # Optimal around 100 characters
            confidence_score = max(0, confidence + 1) / 2  # Normalize confidence
            
            # Check for interesting content
            content_score = self._calculate_content_score(segment['text'])
            
            # Combined score
            total_score = (duration_score * 0.3 + 
                          length_score * 0.2 + 
                          confidence_score * 0.2 + 
                          content_score * 0.3)
            
            scored_segments.append({
                **segment,
                'score': total_score,
                'duration_s': duration_s
            })
        
        return scored_segments
    
    def _calculate_content_score(self, text: str) -> float:
        """Score content based on interesting keywords and patterns"""
        text_lower = text.lower()
        
        # Interesting keywords that suggest engaging content
        interesting_keywords = [
            'important', 'significant', 'amazing', 'incredible', 'breakthrough',
            'discover', 'research', 'study', 'found', 'show', 'prove',
            'question', 'answer', 'problem', 'solution', 'key', 'secret'
        ]
        
        # Count interesting keywords
        keyword_count = sum(1 for keyword in interesting_keywords if keyword in text_lower)
        
        # Bonus for questions (engaging)
        question_bonus = 0.2 if '?' in text else 0
        
        # Bonus for complete thoughts
        complete_bonus = 0.1 if text.strip().endswith(('.', '!', '?')) else 0
        
        return min(1.0, (keyword_count * 0.1) + question_bonus + complete_bonus)
    
    def _has_complete_sentence(self, text: str) -> bool:
        """Check if text contains complete sentences"""
        text = text.strip()
        return len(text) > 10 and any(text.endswith(punct) for punct in '.!?')
    
    def _remove_overlaps(self, segments: list) -> list:
        """Remove overlapping segments, keeping higher scored ones"""
        if not segments:
            return []
        
        # Sort by score
        segments.sort(key=lambda x: x['score'], reverse=True)
        
        non_overlapping = []
        
        for segment in segments:
            overlaps = False
            
            for existing in non_overlapping:
                if self._segments_overlap(segment, existing):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(segment)
        
        return non_overlapping
    
    def _segments_overlap(self, seg1: dict, seg2: dict) -> bool:
        """Check if two segments overlap in time"""
        return not (seg1['end_ms'] <= seg2['start_ms'] or seg2['end_ms'] <= seg1['start_ms'])
```

**Features**:
- Quality-based filtering (duration, text length, confidence)
- Multi-factor scoring system
- Content analysis for interesting keywords
- Overlap removal
- Complete sentence detection

**Deliverable**: Intelligent segment selection system
**Time**: 4 hours

---

## 📋 PHASE 2: CORE FEATURES (Tasks 6-10)

### Task 6: Professional Video Transitions [MEDIUM PRIORITY]
**Objective**: Add smooth transitions between segments

**Implementation**:
- Crossfade transitions between clips
- Fade in/out at beginning and end
- Audio ducking during transitions
- Configurable transition duration

**Time**: 3 hours

### Task 7: Error Handling & Recovery [MEDIUM PRIORITY]
**Objective**: Comprehensive error handling for production use

**Implementation**:
- Try-catch blocks around all external calls
- Graceful degradation when services fail
- Detailed error logging
- Automatic retry mechanisms
- Validation of all inputs and outputs

**Time**: 4 hours

### Task 8: Progress Tracking & Logging [MEDIUM PRIORITY]
**Objective**: Real-time progress feedback

**Implementation**:
- Progress bars for long operations
- Detailed logging at multiple levels
- Performance metrics collection
- Status callbacks for UI integration

**Time**: 2 hours

### Task 9: Configuration Management [MEDIUM PRIORITY]
**Objective**: Proper configuration system

**Implementation**:
- YAML/JSON configuration files
- Environment variable support
- Configuration validation
- Default value management

**Time**: 2 hours

### Task 10: Comprehensive Testing [MEDIUM PRIORITY]
**Objective**: Test all working components

**Implementation**:
- Unit tests for all core functions
- Integration tests for full pipeline
- Performance benchmarks
- Error scenario testing

**Time**: 6 hours

---

## 📋 PHASE 3: PRODUCTION READY (Tasks 11-14)

### Task 11: Clean CLI Interface [MEDIUM PRIORITY]
**Objective**: Professional command-line interface

**Implementation**:
- Argument parsing with argparse
- Help documentation
- Input validation
- Output formatting

**Time**: 3 hours

### Task 12: Performance Optimization [LOW PRIORITY]
**Objective**: Optimize for speed and efficiency

**Implementation**:
- Memory usage optimization
- Parallel processing where possible
- Caching strategies
- Resource cleanup

**Time**: 4 hours

### Task 13: Documentation [LOW PRIORITY]
**Objective**: Complete documentation for real features

**Implementation**:
- API documentation
- Usage examples
- Installation guide
- Troubleshooting guide

**Time**: 3 hours

### Task 14: End-to-End Testing [HIGH PRIORITY]
**Objective**: Final validation with real videos

**Implementation**:
- Test with various video formats
- Test with different lengths
- Test error scenarios
- Performance benchmarking

**Time**: 4 hours

---

## 🎯 IMPLEMENTATION SCHEDULE

### Week 1: Foundation (Tasks 1-5)
- **Day 1-2**: Tasks 1-2 (Cleanup + Transcription) - 6 hours
- **Day 3-4**: Tasks 3-4 (Face Detection + Video Processing) - 14 hours  
- **Day 5**: Task 5 (Segment Selection) - 4 hours

### Week 2: Core Features (Tasks 6-10)
- **Day 1**: Tasks 6-7 (Transitions + Error Handling) - 7 hours
- **Day 2**: Tasks 8-9 (Progress + Config) - 4 hours
- **Day 3**: Task 10 (Testing) - 6 hours

### Week 3: Production (Tasks 11-14)
- **Day 1**: Task 11 (CLI) - 3 hours
- **Day 2**: Tasks 12-13 (Optimization + Docs) - 7 hours
- **Day 3**: Task 14 (End-to-End Testing) - 4 hours

**Total Effort**: ~60 hours over 3 weeks

---

## 🎯 SUCCESS CRITERIA

### Technical Milestones:
1. ✅ 100% working transcription (no fake data)
2. ✅ 100% working face detection and cropping
3. ✅ 100% working video processing pipeline
4. ✅ Professional output quality (1080x1920, smooth playback)
5. ✅ Error-free operation on test videos
6. ✅ Complete documentation and testing

### Quality Gates:
- No fake/mock functions remaining
- All tests passing
- Professional output quality
- Production-ready error handling
- Complete documentation

---

## 💡 RISK MITIGATION

### High-Risk Areas:
1. **FFmpeg Compatibility**: Test on multiple systems
2. **Memory Usage**: Monitor for large videos
3. **Dependency Issues**: Pin all package versions
4. **Performance**: Benchmark on target hardware

### Contingency Plans:
- Fallback transcription methods if Whisper fails
- Center crop fallback if face detection fails
- Quality degradation options for performance
- Comprehensive error messages for debugging

---

This plan provides 100% real functionality with no fake components, focusing on what actually works: transcription, face detection, and video processing. Each task has clear deliverables and time estimates for complete implementation.