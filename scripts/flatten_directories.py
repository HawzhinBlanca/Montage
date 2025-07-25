#!/usr/bin/env python3
"""
Plan and execute directory flattening
"""

import os
import shutil
from pathlib import Path

def analyze_directory_structure():
    """Analyze current directory structure and plan flattening"""
    
    moves = []
    
    # 1. Move AI components to core
    moves.extend([
        {
            "from": "montage/ai/director.py",
            "to": "montage/core/ai_director.py",
            "reason": "AI director is a core component"
        }
    ])
    
    # 2. Move vision tracker to core  
    moves.extend([
        {
            "from": "montage/vision/tracker.py",
            "to": "montage/core/visual_tracker.py",
            "reason": "Visual tracking is a core feature"
        }
    ])
    
    # 3. Move jobs/tasks to api where celery_app was
    moves.extend([
        {
            "from": "montage/jobs/tasks.py",
            "to": "montage/api/celery_tasks.py",
            "reason": "Consolidate Celery components in API"
        }
    ])
    
    # 4. Move pipeline files to core
    moves.extend([
        {
            "from": "montage/pipeline/fast_mode.py",
            "to": "montage/core/fast_pipeline.py",
            "reason": "Pipeline modes belong in core"
        },
        {
            "from": "montage/pipeline/smart_editor.py", 
            "to": "montage/core/smart_editor.py",
            "reason": "Smart editor is core functionality"
        }
    ])
    
    # 5. Directories to remove after moving
    empty_dirs = [
        "montage/ai",
        "montage/vision", 
        "montage/jobs",
        "montage/pipeline",
        "montage/output"  # Empty directory
    ]
    
    return moves, empty_dirs

def update_imports(moves):
    """Generate sed commands to update imports"""
    
    import_updates = []
    
    for move in moves:
        old_module = move["from"].replace("montage/", "").replace(".py", "").replace("/", ".")
        new_module = move["to"].replace("montage/", "").replace(".py", "").replace("/", ".")
        
        import_updates.append({
            "old": f"from montage.{old_module}",
            "new": f"from montage.{new_module}",
            "pattern": f"s/from montage\\.{old_module}/from montage.{new_module}/g"
        })
        
        import_updates.append({
            "old": f"import montage.{old_module}",
            "new": f"import montage.{new_module}",
            "pattern": f"s/import montage\\.{old_module}/import montage.{new_module}/g"
        })
        
        # Handle relative imports
        if "ai/director" in move["from"]:
            import_updates.append({
                "old": "from ..vision.tracker",
                "new": "from .visual_tracker",
                "pattern": "s/from \\.\\.\\.vision\\.tracker/from .visual_tracker/g"
            })
    
    return import_updates

def generate_move_script(moves, empty_dirs, import_updates):
    """Generate shell script to perform the flattening"""
    
    script = """#!/bin/bash
# Directory flattening script
set -e

echo "🚀 Starting directory flattening..."

# Step 1: Move files
"""
    
    for move in moves:
        script += f"""
echo "Moving {move['from']} -> {move['to']}"
git mv {move['from']} {move['to']} || mv {move['from']} {move['to']}
"""
    
    script += """
# Step 2: Update imports
echo "📝 Updating imports..."
"""
    
    for update in import_updates:
        script += f"""
# Update: {update['old']} -> {update['new']}
find montage -name "*.py" -type f -exec sed -i '' '{update['pattern']}' {{}} \\;
"""
    
    script += """
# Step 3: Remove empty directories
echo "🗑️  Removing empty directories..."
"""
    
    for dir_path in empty_dirs:
        script += f"""
rmdir {dir_path} 2>/dev/null || echo "  {dir_path} not empty or already removed"
"""
    
    script += """
echo "✅ Directory flattening complete!"
"""
    
    return script

def main():
    """Generate directory flattening plan"""
    
    print("📊 Analyzing directory structure...")
    
    moves, empty_dirs = analyze_directory_structure()
    import_updates = update_imports(moves)
    
    print(f"\n📋 Flattening Plan:")
    print(f"   Files to move: {len(moves)}")
    print(f"   Import updates: {len(import_updates)}")
    print(f"   Directories to remove: {len(empty_dirs)}")
    
    print("\n🚚 File Moves:")
    for move in moves:
        print(f"   {move['from']}")
        print(f"   → {move['to']}")
        print(f"   Reason: {move['reason']}\n")
    
    # Generate script
    script = generate_move_script(moves, empty_dirs, import_updates)
    
    with open("flatten_dirs.sh", "w") as f:
        f.write(script)
    
    os.chmod("flatten_dirs.sh", 0o755)
    
    print("✅ Generated flatten_dirs.sh")
    print("   Review and run: ./flatten_dirs.sh")

if __name__ == "__main__":
    main()