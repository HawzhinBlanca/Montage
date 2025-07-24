#!/bin/bash
# Tasks.md Step 2: Metal cache purge script
# Clears GPU memory cache hourly to prevent memory buildup

# Log execution
echo "[$(date)] Starting Metal cache purge" >> ~/Library/Logs/metal-cache-purge.log

# Clear PyTorch GPU cache if Python/PyTorch available
if command -v python3 &> /dev/null; then
    python3 -c "
try:
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print('PyTorch MPS cache cleared')
except:
    pass
" >> ~/Library/Logs/metal-cache-purge.log 2>&1
fi

# Clear system GPU memory pressure
# This forces macOS to reclaim unused GPU memory
memory_pressure -l critical

# Log completion
echo "[$(date)] Metal cache purge completed" >> ~/Library/Logs/metal-cache-purge.log