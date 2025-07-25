#!/usr/bin/env python3
"""
Identify files safe to delete based on multiple criteria
"""

import ast
import subprocess
from pathlib import Path


def get_test_coverage():
    """Get coverage data for all files"""
    print("📊 Analyzing test coverage...")
    coverage_data = {}

    try:
        # Run coverage report
        subprocess.run(["coverage", "run", "-m", "pytest", "-q"],
                      capture_output=True, check=False)
        result = subprocess.run(["coverage", "report", "--format=total"],
                               capture_output=True, text=True)

        # Parse coverage output
        for line in result.stdout.split('\n'):
            if line and '.py' in line and not line.startswith('TOTAL'):
                parts = line.split()
                if len(parts) >= 2:
                    file_path = parts[0]
                    try:
                        coverage_pct = int(parts[-1].rstrip('%'))
                        coverage_data[file_path] = coverage_pct
                    except ValueError:
                        pass
    except Exception:
        print("   ⚠️  Coverage data not available")

    return coverage_data

def parse_vulture_output():
    """Parse vulture findings"""
    unused_files = set()

    if Path("vulture_raw.txt").exists():
        with open("vulture_raw.txt") as f:
            for line in f:
                if line.strip() and ':' in line:
                    file_path = line.split(':')[0]
                    unused_files.add(file_path)

    return unused_files

def analyze_imports():
    """Find files with zero imports from other modules"""
    imported_files = set()

    for py_file in Path("montage").rglob("*.py"):
        try:
            with open(py_file) as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith('montage'):
                        # Convert module to file path
                        module_path = node.module.replace('.', '/') + '.py'
                        imported_files.add(f"montage/{module_path}")
        except Exception:
            pass

    return imported_files

def get_commit_count():
    """Get commit counts from code_churn.csv"""
    commit_counts = {}

    if Path("code_churn.csv").exists():
        with open("code_churn.csv") as f:
            next(f)  # Skip header
            for line in f:
                if ',' in line:
                    file_path, count = line.strip().split(',')
                    commit_counts[file_path] = int(count)

    return commit_counts

def check_adr_references():
    """Check if files are referenced in ADRs"""
    referenced_files = set()

    adr_dir = Path("docs/adr")
    if adr_dir.exists():
        for adr_file in adr_dir.glob("*.md"):
            with open(adr_file) as f:
                content = f.read()
                # Look for file references
                for py_file in Path("montage").rglob("*.py"):
                    if str(py_file) in content or py_file.name in content:
                        referenced_files.add(str(py_file))

    return referenced_files

def identify_deletable_files():
    """Identify files safe to delete"""
    print("\n🔍 Analyzing files for safe deletion...")

    # Gather all data
    coverage_data = get_test_coverage()
    vulture_files = parse_vulture_output()
    imported_files = analyze_imports()
    commit_counts = get_commit_count()
    adr_files = check_adr_references()

    # Critical files to never delete
    critical_patterns = [
        "__init__.py",
        "conftest.py",
        "setup.py",
        "settings",
        "config",
        "auth",
        "security",
        "db.py",
        "metrics.py"
    ]

    deletable = []

    for py_file in Path("montage").rglob("*.py"):
        file_str = str(py_file)

        # Skip critical files
        if any(pattern in file_str for pattern in critical_patterns):
            continue

        # Check all criteria
        criteria = {
            "zero_coverage": coverage_data.get(file_str, 0) == 0,
            "vulture_unused": file_str in vulture_files,
            "no_imports": file_str not in imported_files,
            "low_churn": commit_counts.get(file_str, 0) < 3,
            "not_in_adr": file_str not in adr_files
        }

        # File is deletable if ALL criteria are met
        if all(criteria.values()):
            deletable.append({
                "file": file_str,
                "coverage": coverage_data.get(file_str, 0),
                "commits": commit_counts.get(file_str, 0)
            })

    return deletable

def main():
    """Identify and report deletable files"""
    deletable = identify_deletable_files()

    print(f"\n📋 Found {len(deletable)} files safe to delete:")

    # Write report
    with open("deletable_files.txt", "w") as f:
        f.write("# Files Safe to Delete\n")
        f.write("# All meet criteria: zero coverage, unused by vulture, no imports, <3 commits, not in ADRs\n\n")

        for item in sorted(deletable, key=lambda x: x['file']):
            print(f"   - {item['file']} (coverage: {item['coverage']}%, commits: {item['commits']})")
            f.write(f"{item['file']}\n")

    print("\n💾 Report saved to: deletable_files.txt")

    # Create deletion script
    with open("delete_dead_files.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Script to delete identified dead files\n")
        f.write("# Review carefully before running!\n\n")
        f.write("set -e\n\n")

        for item in deletable:
            f.write(f"echo \"Deleting {item['file']}...\"\n")
            f.write(f"git rm {item['file']}\n")

        f.write("\necho \"Deleted {len(deletable)} files\"\n")

    subprocess.run(["chmod", "+x", "delete_dead_files.sh"])
    print("✅ Created delete_dead_files.sh (review before running)")

if __name__ == "__main__":
    main()
