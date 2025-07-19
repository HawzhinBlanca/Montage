# Montage - AI-Powered Video Processing Pipeline

Professional video highlight extraction using real AI intelligence.

## Features

- **AI Intelligence**: Real content analysis using Deepgram + OpenAI
- **Vertical Format**: Automatic 1080x1920 output for social media
- **Smart Highlights**: AI selects the most engaging segments
- **Multi-Provider**: Fallback support for reliability
- **Professional Quality**: Broadcast-ready encoding

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

- `src/analyze_video.py` - Multi-provider transcription & analysis
- `src/highlight_selector.py` - AI-powered highlight selection
- `src/run_pipeline.py` - Main processing pipeline
- `src/ffmpeg_utils.py` - Video processing utilities
- `src/resolve_mcp.py` - DaVinci Resolve integration (optional)

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