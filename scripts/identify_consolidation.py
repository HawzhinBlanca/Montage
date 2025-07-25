#!/usr/bin/env python3
"""
Identify utilities that can be consolidated
"""

import ast
from pathlib import Path
from collections import defaultdict

def analyze_utility_similarities():
    """Find utilities with similar functionality"""
    
    # Categories of utilities to consolidate
    categories = {
        "logging": [],
        "memory_management": [],
        "resource_management": [],
        "error_handling": [],
        "validation": [],
        "process_management": [],
    }
    
    # Patterns to identify categories
    patterns = {
        "logging": ["logger", "logging", "log_", "get_logger", "formatter"],
        "memory_management": ["memory", "mem_", "oom", "Memory"],
        "resource_management": ["resource", "cleanup", "tracker", "Resource"],
        "error_handling": ["error", "exception", "Error", "Exception", "handle_"],
        "validation": ["validate", "validator", "check_", "Validator"],
        "process_management": ["process", "Process", "subprocess", "ffmpeg_process"],
    }
    
    # Scan utilities
    utils_dir = Path("montage/utils")
    core_dir = Path("montage/core")
    
    for py_file in list(utils_dir.glob("*.py")) + list(core_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
            
        try:
            with open(py_file) as f:
                content = f.read()
                tree = ast.parse(content)
                
            # Check against patterns
            for category, keywords in patterns.items():
                for keyword in keywords:
                    if keyword in content:
                        categories[category].append(str(py_file))
                        break
                        
        except Exception as e:
            print(f"Error analyzing {py_file}: {e}")
    
    return categories

def find_import_overlaps():
    """Find files that import similar modules"""
    
    imports_by_file = defaultdict(set)
    
    for py_file in Path("montage").rglob("*.py"):
        try:
            with open(py_file) as f:
                tree = ast.parse(f.read())
                
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports_by_file[str(py_file)].add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports_by_file[str(py_file)].add(node.module)
        except:
            pass
    
    # Find files with similar imports
    similar_imports = defaultdict(list)
    files = list(imports_by_file.keys())
    
    for i, file1 in enumerate(files):
        for file2 in files[i+1:]:
            imports1 = imports_by_file[file1]
            imports2 = imports_by_file[file2]
            
            overlap = imports1 & imports2
            if len(overlap) > 5:  # Significant overlap
                similar_imports[f"{file1} <-> {file2}"].append(len(overlap))
    
    return similar_imports

def main():
    """Analyze and report consolidation opportunities"""
    
    print("🔍 Analyzing utilities for consolidation...\n")
    
    # Analyze by category
    categories = analyze_utility_similarities()
    
    consolidation_opportunities = []
    
    print("📊 Utilities by Category:")
    for category, files in categories.items():
        if len(files) > 1:
            print(f"\n{category.upper()} ({len(files)} files):")
            for file in sorted(files):
                print(f"  - {file}")
            
            # Mark as consolidation opportunity
            if len(files) > 2:
                consolidation_opportunities.append({
                    "category": category,
                    "files": files,
                    "count": len(files)
                })
    
    # Find specific duplicates
    print("\n📋 Specific Consolidation Opportunities:")
    
    # 1. Logging utilities
    logging_files = [
        "montage/utils/logging_config.py",
        "montage/utils/secure_logging.py"
    ]
    if all(Path(f).exists() for f in logging_files):
        print("\n1. LOGGING: Merge secure_logging into logging_config")
        print("   - montage/utils/secure_logging.py -> montage/utils/logging_config.py")
        consolidation_opportunities.append({
            "action": "merge",
            "source": "montage/utils/secure_logging.py",
            "target": "montage/utils/logging_config.py",
            "reason": "Combine all logging functionality"
        })
    
    # 2. Memory management
    memory_files = [
        "montage/utils/memory_manager.py",
        "montage/utils/memory_init.py",
        "montage/utils/resource_manager.py"
    ]
    existing_memory = [f for f in memory_files if Path(f).exists()]
    if len(existing_memory) > 1:
        print("\n2. MEMORY: Consolidate memory/resource management")
        print(f"   - Merge {len(existing_memory)} files into montage/utils/memory_manager.py")
        for f in existing_memory[1:]:
            consolidation_opportunities.append({
                "action": "merge",
                "source": f,
                "target": existing_memory[0],
                "reason": "Unified memory/resource management"
            })
    
    # 3. Process management
    process_files = [
        "montage/utils/ffmpeg_process_manager.py",
        "montage/utils/ffmpeg_utils.py"
    ]
    if all(Path(f).exists() for f in process_files):
        print("\n3. PROCESS: Merge FFmpeg utilities")
        print("   - montage/utils/ffmpeg_process_manager.py -> montage/utils/ffmpeg_utils.py")
        consolidation_opportunities.append({
            "action": "merge",
            "source": "montage/utils/ffmpeg_process_manager.py",
            "target": "montage/utils/ffmpeg_utils.py",
            "reason": "Consolidate FFmpeg functionality"
        })
    
    # Write consolidation plan
    with open("consolidation_plan.txt", "w") as f:
        f.write("# Utility Consolidation Plan\n\n")
        
        for item in consolidation_opportunities:
            if "action" in item:
                f.write(f"## {item['reason']}\n")
                f.write(f"Action: {item['action']}\n")
                f.write(f"Source: {item['source']}\n")
                f.write(f"Target: {item['target']}\n\n")
            else:
                f.write(f"## {item['category'].upper()}\n")
                f.write(f"Files to consolidate ({item['count']}):\n")
                for file in item['files']:
                    f.write(f"- {file}\n")
                f.write("\n")
    
    print(f"\n✅ Analysis complete!")
    print(f"📄 Consolidation plan saved to: consolidation_plan.txt")
    print(f"🎯 Found {len(consolidation_opportunities)} consolidation opportunities")

if __name__ == "__main__":
    main()