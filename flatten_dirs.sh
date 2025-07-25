#!/bin/bash
# Directory flattening script
set -e

echo "🚀 Starting directory flattening..."

# Step 1: Move files

echo "Moving montage/ai/director.py -> montage/core/ai_director.py"
git mv montage/ai/director.py montage/core/ai_director.py || mv montage/ai/director.py montage/core/ai_director.py

echo "Moving montage/vision/tracker.py -> montage/core/visual_tracker.py"
git mv montage/vision/tracker.py montage/core/visual_tracker.py || mv montage/vision/tracker.py montage/core/visual_tracker.py

echo "Moving montage/jobs/tasks.py -> montage/api/celery_tasks.py"
git mv montage/jobs/tasks.py montage/api/celery_tasks.py || mv montage/jobs/tasks.py montage/api/celery_tasks.py

echo "Moving montage/pipeline/fast_mode.py -> montage/core/fast_pipeline.py"
git mv montage/pipeline/fast_mode.py montage/core/fast_pipeline.py || mv montage/pipeline/fast_mode.py montage/core/fast_pipeline.py

echo "Moving montage/pipeline/smart_editor.py -> montage/core/smart_editor.py"
git mv montage/pipeline/smart_editor.py montage/core/smart_editor.py || mv montage/pipeline/smart_editor.py montage/core/smart_editor.py

# Step 2: Update imports
echo "📝 Updating imports..."

# Update: from montage.ai.director -> from montage.core.ai_director
find montage -name "*.py" -type f -exec sed -i '' 's/from montage\.ai.director/from montage.core.ai_director/g' {} \;

# Update: import montage.ai.director -> import montage.core.ai_director
find montage -name "*.py" -type f -exec sed -i '' 's/import montage\.ai.director/import montage.core.ai_director/g' {} \;

# Update: from ..vision.tracker -> from .visual_tracker
find montage -name "*.py" -type f -exec sed -i '' 's/from \.\.\.vision\.tracker/from .visual_tracker/g' {} \;

# Update: from montage.vision.tracker -> from montage.core.visual_tracker
find montage -name "*.py" -type f -exec sed -i '' 's/from montage\.vision.tracker/from montage.core.visual_tracker/g' {} \;

# Update: import montage.vision.tracker -> import montage.core.visual_tracker
find montage -name "*.py" -type f -exec sed -i '' 's/import montage\.vision.tracker/import montage.core.visual_tracker/g' {} \;

# Update: from montage.jobs.tasks -> from montage.api.celery_tasks
find montage -name "*.py" -type f -exec sed -i '' 's/from montage\.jobs.tasks/from montage.api.celery_tasks/g' {} \;

# Update: import montage.jobs.tasks -> import montage.api.celery_tasks
find montage -name "*.py" -type f -exec sed -i '' 's/import montage\.jobs.tasks/import montage.api.celery_tasks/g' {} \;

# Update: from montage.pipeline.fast_mode -> from montage.core.fast_pipeline
find montage -name "*.py" -type f -exec sed -i '' 's/from montage\.pipeline.fast_mode/from montage.core.fast_pipeline/g' {} \;

# Update: import montage.pipeline.fast_mode -> import montage.core.fast_pipeline
find montage -name "*.py" -type f -exec sed -i '' 's/import montage\.pipeline.fast_mode/import montage.core.fast_pipeline/g' {} \;

# Update: from montage.pipeline.smart_editor -> from montage.core.smart_editor
find montage -name "*.py" -type f -exec sed -i '' 's/from montage\.pipeline.smart_editor/from montage.core.smart_editor/g' {} \;

# Update: import montage.pipeline.smart_editor -> import montage.core.smart_editor
find montage -name "*.py" -type f -exec sed -i '' 's/import montage\.pipeline.smart_editor/import montage.core.smart_editor/g' {} \;

# Step 3: Remove empty directories
echo "🗑️  Removing empty directories..."

rmdir montage/ai 2>/dev/null || echo "  montage/ai not empty or already removed"

rmdir montage/vision 2>/dev/null || echo "  montage/vision not empty or already removed"

rmdir montage/jobs 2>/dev/null || echo "  montage/jobs not empty or already removed"

rmdir montage/pipeline 2>/dev/null || echo "  montage/pipeline not empty or already removed"

rmdir montage/output 2>/dev/null || echo "  montage/output not empty or already removed"

echo "✅ Directory flattening complete!"
