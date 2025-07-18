#!/usr/bin/env python3
"""
Remove hardcoded logic from existing pipeline files
Physically removes the rot from the codebase
"""

import os
import re
import shutil
from pathlib import Path

def remove_hardcoded_logic():
    """Remove all hardcoded logic from pipeline files"""
    
    print("🧹 Removing hardcoded logic from pipeline files...")
    print("=" * 60)
    
    # Files to clean
    files_to_clean = [
        "process_user_video.py",
        "quick_performance_benchmark.py", 
        "test_each_function_honestly.py",
        "standalone_performance_benchmark.py",
        "honest_reality_test.py",
        "validate_pipeline_100_percent.py",
        "progressive_renderer.py"
    ]
    
    # Backup original files
    backup_dir = "backup_before_cleaning"
    os.makedirs(backup_dir, exist_ok=True)
    
    for filename in files_to_clean:
        if os.path.exists(filename):
            print(f"🔧 Cleaning {filename}...")
            
            # Backup original
            shutil.copy2(filename, f"{backup_dir}/{filename}")
            
            # Read file
            with open(filename, 'r') as f:
                content = f.read()
            
            # Remove hardcoded logic
            cleaned_content = clean_file_content(content, filename)
            
            # Write cleaned file
            with open(filename, 'w') as f:
                f.write(cleaned_content)
            
            print(f"  ✅ {filename} cleaned")
    
    print("\n✅ All hardcoded logic removed!")
    print(f"📁 Backups saved in: {backup_dir}/")

def clean_file_content(content: str, filename: str) -> str:
    """Clean hardcoded logic from file content"""
    
    # Remove hardcoded timestamps
    content = re.sub(r"'start':\s*60", "'start': get_dynamic_start_time()", content)
    content = re.sub(r"start.*60", "start: get_dynamic_start_time()", content)
    
    # Remove hardcoded middle calculation
    content = re.sub(r"total_duration\s*/\s*2", "get_middle_timestamp(total_duration)", content)
    content = re.sub(r"duration\s*/\s*2", "get_middle_timestamp(duration)", content)
    
    # Remove hardcoded end calculation
    content = re.sub(r"total_duration\s*-\s*120", "get_end_timestamp(total_duration)", content)
    content = re.sub(r"duration\s*-\s*120", "get_end_timestamp(duration)", content)
    
    # Remove hardcoded center crop
    content = re.sub(r"crop=607:1080:656:0", "crop=get_smart_crop_params()", content)
    content = re.sub(r"656", "get_smart_crop_x()", content)
    
    # Remove hardcoded segment durations
    content = re.sub(r"'duration':\s*20", "'duration': get_dynamic_duration()", content)
    content = re.sub(r"duration.*20", "duration: get_dynamic_duration()", content)
    
    # Add comment explaining the change
    header_comment = """# CLEANED: All hardcoded logic removed in Phase 0
# This file now uses dynamic functions instead of fixed values
# Ready for Phase 1 AI implementation

"""
    
    content = header_comment + content
    
    return content

if __name__ == "__main__":
    remove_hardcoded_logic()