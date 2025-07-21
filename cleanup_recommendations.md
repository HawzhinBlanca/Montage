# Montage Project Cleanup Recommendations

## 🎯 Summary: What to Keep vs Remove

Based on the comprehensive audit, here's what you should do:

## ✅ KEEP THESE (100% Working)
```
src/core/analyze_video.py          # Real Whisper + Deepgram transcription
src/providers/video_processor.py   # Real FFmpeg processing
src/providers/audio_normalizer.py  # Real EBU R128 normalization
src/core/db.py                     # Real PostgreSQL operations
src/core/checkpoint.py             # Real Redis checkpointing
src/core/cost.py                   # Real budget enforcement
src/utils/ffmpeg_utils.py          # Real video utilities
src/utils/intelligent_crop.py      # Real face detection (OpenCV)
```

## 🔴 REMOVE THESE (Fake/Overhyped)
```
src/core/emotion_analyzer.py       # Just keyword matching, not "advanced AI"
src/core/narrative_detector.py     # Just regex patterns, not "narrative beats"
src/core/speaker_analysis.py       # Basic heuristics, not ML analysis
src/utils/video_effects.py         # Mostly empty stubs
```

## 🟡 FIX THESE (Partially Working)
```
src/core/highlight_selector.py     # Keep Gemini API, remove overhyped claims
src/providers/resolve_mcp.py       # Keep DaVinci integration, fix fallbacks
```

## 📝 Specific Actions

### 1. Honest Feature Description
Instead of:
- ❌ "Advanced AI emotion mapping"
- ❌ "Narrative beat detection with story intelligence"
- ❌ "Multi-speaker ML analysis"

Use:
- ✅ "Keyword-based content analysis"
- ✅ "Pattern matching for story segments"
- ✅ "Basic speaker role detection"

### 2. Core Working Pipeline
Your real value proposition is:
```
Real Audio Transcription → Real Face Detection → Real Video Processing
```

### 3. Clean Dependencies
Remove unused AI/ML dependencies that aren't actually used:
```bash
# Remove these if not genuinely used
pip uninstall transformers torch torchvision pyannote-audio
```

### 4. Simplified Architecture
Focus on what actually works:
```
Input Video → Whisper Transcription → Face-based Cropping → FFmpeg Processing → Output
```

## 🎯 Recommended Next Steps

1. **Create clean entry point** using only real functions
2. **Remove fake AI terminology** from documentation
3. **Focus on core value**: Real transcription + Real video processing
4. **Test only working components**
5. **Honest marketing**: Position as practical video processing tool

## 💡 What Makes Your Project Actually Valuable

Don't sell fake AI features. Your real value is:
- ✅ Reliable Whisper transcription 
- ✅ Face-detection based cropping
- ✅ Professional FFmpeg processing
- ✅ Cost-controlled pipeline
- ✅ Database-backed operations

This is solid, practical functionality that actually works.

## 🚀 Cleaned Up Project Structure

```
montage/
├── core/
│   ├── transcription.py     # Real Whisper/Deepgram
│   ├── video_processing.py  # Real FFmpeg operations
│   ├── face_detection.py    # Real OpenCV face detection
│   ├── database.py          # Real PostgreSQL operations
│   └── cost_tracking.py     # Real budget enforcement
├── utils/
│   ├── ffmpeg_helpers.py    # Real video utilities
│   └── file_operations.py   # Real file handling
└── main.py                  # Clean entry point
```

## ⚡ Bottom Line

**Stop pretending to be an AI company. Become an excellent video processing company.**

Your real technical skills are in:
- Video processing pipelines
- Audio transcription integration  
- Computer vision (face detection)
- Database operations
- Cost management

These are valuable, marketable skills. Build on them honestly.