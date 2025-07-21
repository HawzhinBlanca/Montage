# Montage - Professional Video Processing Pipeline

Production-ready video processing system with real transcription and intelligent cropping.

## Features

- **Real Transcription**: Whisper + Deepgram ASR with word-level timestamps
- **Intelligent Cropping**: OpenCV face detection for optimal vertical format
- **Professional Processing**: High-performance FFmpeg pipeline with FIFO streaming
- **Budget Control**: Hard cost caps with real-time tracking
- **Production Ready**: Comprehensive error handling, metrics, and monitoring

## Quick Start

```bash
# Basic usage - extract AI highlights
python run_montage.py video.mp4

# Create vertical format for social media
python run_montage.py video.mp4 --vertical

# Show video info only
python run_montage.py video.mp4 --info
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure API keys in `.env`:
   ```
   OPENAI_API_KEY=your_key_here
   DEEPGRAM_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here  # Optional
   ```

## Core Components

- `src/core/analyze_video.py` - Real Whisper/Deepgram transcription with ROVER merging
- `src/core/highlight_selector.py` - Content analysis and segment selection
- `src/providers/video_processor.py` - High-performance FIFO-based video processing
- `src/utils/intelligent_crop.py` - OpenCV face detection and smart cropping
- `src/core/metrics.py` - Comprehensive performance monitoring and tracking
- `src/core/cost.py` - Budget enforcement with hard caps

## Output

Processed videos are saved to the `output/` directory with:
- AI-selected highlights
- Professional encoding (H.264, AAC)
- Optional vertical format (1080x1920)
- Metadata reports

## API Providers

- **Deepgram**: Speech-to-text transcription
- **OpenAI**: Content analysis and scoring
- **Claude**: Fallback analysis (optional)
- **Whisper**: Local transcription fallback

## Performance

- 43-minute video → 2-minute highlights
- Real AI analysis (not word counting)
- Intelligent content scoring (1-10)
- Professional quality output

## License

Private project - All rights reserved