#!/bin/bash
#
# Adaptive Quality Pipeline - Quick Run Script
# Creates AI-powered video montages with DaVinci Resolve
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
VIDEO_PATH=""
OUTPUT_DIR="output"
PROJECT_NAME=""
MAX_HIGHLIGHTS=5
USE_PREMIUM_AI=true
REQUIRE_APPROVAL=false

# Function to display usage
usage() {
    echo "Usage: $0 <video_path> [options]"
    echo ""
    echo "Options:"
    echo "  -o, --output DIR          Output directory (default: output)"
    echo "  -n, --name NAME          Project name (auto-generated if not specified)"
    echo "  -h, --highlights NUM     Maximum highlights (default: 5)"
    echo "  --no-ai                  Disable premium AI scoring"
    echo "  --approve                Require human approval"
    echo "  --help                   Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 my_video.mp4 -n \"My Amazing Montage\" -h 7"
    exit 1
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    usage
fi

VIDEO_PATH=$1
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        -h|--highlights)
            MAX_HIGHLIGHTS="$2"
            shift 2
            ;;
        --no-ai)
            USE_PREMIUM_AI=false
            shift
            ;;
        --approve)
            REQUIRE_APPROVAL=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate video file
if [ ! -f "$VIDEO_PATH" ]; then
    echo -e "${RED}Error: Video file not found: $VIDEO_PATH${NC}"
    exit 1
fi

# Display banner
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║          ADAPTIVE QUALITY PIPELINE - MONTAGE CREATOR      ║"
echo "║                   AI-Powered Video Magic                  ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Display configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Video: $VIDEO_PATH"
echo "  Output: $OUTPUT_DIR"
if [ -n "$PROJECT_NAME" ]; then
    echo "  Project: $PROJECT_NAME"
fi
echo "  Max Highlights: $MAX_HIGHLIGHTS"
echo "  Premium AI: $USE_PREMIUM_AI"
echo "  Require Approval: $REQUIRE_APPROVAL"
echo ""

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required${NC}"
    exit 1
fi

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}Error: FFmpeg is required${NC}"
    echo "Install with: brew install ffmpeg"
    exit 1
fi

# Check DaVinci Resolve
if [ ! -d "/Applications/DaVinci Resolve/DaVinci Resolve.app" ]; then
    echo -e "${YELLOW}Warning: DaVinci Resolve not found${NC}"
    echo "The pipeline will run but video editing will fail"
    echo "Download from: https://www.blackmagicdesign.com/products/davinciresolve"
fi

# Check if DaVinci Resolve is running
if ! pgrep -x "Resolve" > /dev/null; then
    echo -e "${YELLOW}Starting DaVinci Resolve...${NC}"
    open -a "DaVinci Resolve"
    sleep 5
fi

echo -e "${GREEN}✓ Dependencies checked${NC}"
echo ""

# Build command
CMD="python3 adaptive_quality_pipeline_master.py \"$VIDEO_PATH\" --output \"$OUTPUT_DIR\""

if [ -n "$PROJECT_NAME" ]; then
    CMD="$CMD --name \"$PROJECT_NAME\""
fi

CMD="$CMD --max-highlights $MAX_HIGHLIGHTS"

if [ "$USE_PREMIUM_AI" = false ]; then
    CMD="$CMD --no-premium-ai"
fi

if [ "$REQUIRE_APPROVAL" = true ]; then
    CMD="$CMD --require-approval"
fi

# Run pipeline
echo -e "${BLUE}Starting Adaptive Quality Pipeline...${NC}"
echo -e "${YELLOW}Command: $CMD${NC}"
echo ""

# Execute with proper error handling
if eval $CMD; then
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    SUCCESS! 🎉                            ║${NC}"
    echo -e "${GREEN}║         Your AI-powered montage is ready!                 ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
else
    echo ""
    echo -e "${RED}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                    FAILED ❌                              ║${NC}"
    echo -e "${RED}║      Check the logs for details                           ║${NC}"
    echo -e "${RED}╚═══════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi