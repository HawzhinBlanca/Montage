#\!/bin/bash
# Install Metal cache purge scheduled job

PLIST_FILE="com.montage.metal-cache-purge.plist"
PLIST_PATH="$HOME/Library/LaunchAgents/$PLIST_FILE"

# Copy plist to LaunchAgents
cp "$PLIST_FILE" "$PLIST_PATH"

# Load the job
launchctl load "$PLIST_PATH"

# Verify it's loaded
if launchctl list | grep -q "com.montage.metal-cache-purge"; then
    echo "✅ Metal cache purge job installed and loaded"
    echo "   Will run hourly to clear GPU memory"
    echo "   Logs: ~/Library/Logs/metal-cache-purge.log"
else
    echo "❌ Failed to load Metal cache purge job"
fi
