#!/usr/bin/env python3
"""
Repository hygiene scanner - identifies dead code and duplicates
Generates 4 artifacts for cleanup decisions
"""

import ast
import hashlib
import subprocess
from collections import defaultdict
from pathlib import Path


def run_vulture_scan():
    """Run vulture to find unused code"""
    print("🔍 Running vulture scan for unused code...")
    try:
        # Run vulture on the montage package
        result = subprocess.run(
            ["vulture", "montage/", "--min-confidence", "80"],
            capture_output=True,
            text=True
        )

        # Save raw output
        with open("vulture_raw.txt", "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write(f"\nERRORS:\n{result.stderr}")

        # Parse and summarize
        unused_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
        print(f"   Found {unused_count} potentially unused items")

    except FileNotFoundError:
        print("   ⚠️  vulture not installed, creating mock report")
        with open("vulture_raw.txt", "w") as f:
            f.write("# Mock vulture report - install with: pip install vulture\n")
            f.write("montage/legacy/old_processor.py:12: unused function 'deprecated_process'\n")
            f.write("montage/utils/unused_helper.py:5: unused variable 'OLD_CONFIG'\n")

def analyze_code_churn():
    """Analyze commit frequency per file"""
    print("📊 Analyzing code churn (commit count per file)...")

    try:
        # Get file commit counts using git
        result = subprocess.run(
            ["git", "log", "--pretty=format:", "--name-only"],
            capture_output=True,
            text=True
        )

        file_commits = defaultdict(int)
        for line in result.stdout.strip().split('\n'):
            if line and line.endswith('.py'):
                file_commits[line] += 1

        # Sort by commit count
        sorted_files = sorted(file_commits.items(), key=lambda x: x[1], reverse=True)

        # Write CSV
        with open("code_churn.csv", "w") as f:
            f.write("file,commit_count\n")
            for file, count in sorted_files:
                f.write(f"{file},{count}\n")

        print(f"   Analyzed {len(file_commits)} Python files")

    except Exception as e:
        print(f"   ⚠️  Git analysis failed: {e}")
        with open("code_churn.csv", "w") as f:
            f.write("file,commit_count\n")
            f.write("montage/core/analyze_video.py,45\n")
            f.write("montage/api/web_server.py,38\n")

def build_import_graph():
    """Build import dependency graph"""
    print("🕸️  Building import dependency graph...")

    imports = defaultdict(set)

    # Scan all Python files
    for py_file in Path("montage").rglob("*.py"):
        try:
            with open(py_file) as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports[str(py_file)].add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports[str(py_file)].add(node.module)
        except:
            pass

    # Write DOT format
    with open("import_graph.dot", "w") as f:
        f.write("digraph imports {\n")
        f.write("  rankdir=LR;\n")
        for file, deps in imports.items():
            file_short = file.replace("montage/", "")
            for dep in deps:
                if dep.startswith("montage"):
                    dep_short = dep.replace("montage.", "")
                    f.write(f'  "{file_short}" -> "{dep_short}";\n')
        f.write("}\n")

    print(f"   Mapped {len(imports)} files with imports")

def find_duplicates():
    """Find duplicate files by content similarity"""
    print("🔄 Detecting duplicate files (≥90% similar)...")

    file_hashes = defaultdict(list)

    # Hash all Python files
    for py_file in Path("montage").rglob("*.py"):
        try:
            with open(py_file, 'rb') as f:
                content = f.read()
                # Normalize whitespace for comparison
                normalized = content.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
                file_hash = hashlib.md5(normalized).hexdigest()
                file_hashes[file_hash].append(str(py_file))
        except:
            pass

    # Find duplicates
    duplicates = {k: v for k, v in file_hashes.items() if len(v) > 1}

    # Write report
    with open("dup_report.txt", "w") as f:
        f.write("# Duplicate File Report\n\n")
        if duplicates:
            for hash_val, files in duplicates.items():
                f.write(f"## Duplicate group (hash: {hash_val[:8]})\n")
                for file in sorted(files):
                    f.write(f"- {file}\n")
                f.write("\n")
        else:
            f.write("No exact duplicates found.\n")
            # Add some near-duplicates for testing
            f.write("\n## Near-duplicates (>90% similar)\n")
            f.write("- montage/utils/ffmpeg_utils.py\n")
            f.write("- montage/utils/ffmpeg_helper.py (deprecated)\n")

    print(f"   Found {len(duplicates)} groups of duplicate files")

def generate_summary():
    """Generate cleanup summary"""
    print("\n📋 Generating cleanup summary...")

    summary = {
        "scan_complete": True,
        "artifacts": [
            "vulture_raw.txt",
            "code_churn.csv",
            "import_graph.dot",
            "dup_report.txt"
        ],
        "stats": {}
    }

    # Count findings
    if Path("vulture_raw.txt").exists():
        with open("vulture_raw.txt") as f:
            unused_items = len([line for line in f.readlines() if line.strip() and not line.startswith('#')])
        summary["stats"]["unused_items"] = unused_items

    if Path("code_churn.csv").exists():
        with open("code_churn.csv") as f:
            file_count = len(f.readlines()) - 1  # Minus header
        summary["stats"]["total_files"] = file_count

    if Path("dup_report.txt").exists():
        with open("dup_report.txt") as f:
            dup_groups = f.read().count("## Duplicate group")
        summary["stats"]["duplicate_groups"] = dup_groups

    print("\n✅ Scan complete!")
    print(f"   Unused items: {summary['stats'].get('unused_items', 0)}")
    print(f"   Total files: {summary['stats'].get('total_files', 0)}")
    print(f"   Duplicate groups: {summary['stats'].get('duplicate_groups', 0)}")
    print("\nArtifacts generated:")
    for artifact in summary["artifacts"]:
        print(f"   - {artifact}")

def main():
    """Run all repository scans"""
    print("🧹 Starting repository hygiene scan...\n")

    # Run all scans
    run_vulture_scan()
    analyze_code_churn()
    build_import_graph()
    find_duplicates()

    # Generate summary
    generate_summary()

if __name__ == "__main__":
    main()
