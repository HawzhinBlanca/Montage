#!/usr/bin/env python3
"""Scan for stub implementations (pass, NotImplementedError, TODO) in production code."""

import ast
import os
import sys
from pathlib import Path

def is_stub_function(node):
    """Check if a function is a stub implementation."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    
    # Check for single pass statement
    if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
        return True
    
    # Check for NotImplementedError
    if len(node.body) == 1 and isinstance(node.body[0], ast.Raise):
        raise_node = node.body[0]
        if isinstance(raise_node.exc, ast.Call):
            if hasattr(raise_node.exc.func, 'id') and raise_node.exc.func.id == 'NotImplementedError':
                return True
            if isinstance(raise_node.exc.func, ast.Name) and raise_node.exc.func.id == 'NotImplementedError':
                return True
    
    # Check for TODO comments
    for stmt in ast.walk(node):
        if hasattr(stmt, 'lineno'):
            # This is a simplified check - in reality we'd need to parse comments
            pass
    
    return False

def scan_file(filepath):
    """Scan a single Python file for stubs."""
    stubs = []
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            tree = ast.parse(content)
            
        # Check if this is within a fallback/except block
        in_fallback = False
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if is_stub_function(node):
                # Check if this function is within a fallback implementation
                is_fallback = False
                for i in range(max(0, node.lineno - 10), node.lineno):
                    if i < len(lines) and ('except ImportError' in lines[i] or '# Fallback' in lines[i]):
                        is_fallback = True
                        break
                
                if not is_fallback:
                    stubs.append({
                        'file': str(filepath),
                        'function': node.name,
                        'line': node.lineno,
                        'type': 'stub'
                    })
                
        # Also scan for TODO comments
        for i, line in enumerate(lines, 1):
            if 'TODO' in line or 'FIXME' in line:
                stubs.append({
                    'file': str(filepath),
                    'function': 'N/A',
                    'line': i,
                    'type': 'TODO'
                })
                
    except Exception as e:
        print(f"Error scanning {filepath}: {e}", file=sys.stderr)
    
    return stubs

def main():
    """Scan the montage package for stubs."""
    root = Path(__file__).parent.parent / 'montage'
    all_stubs = []
    
    for py_file in root.rglob('*.py'):
        # Skip test files
        if 'test' in str(py_file) or '__pycache__' in str(py_file):
            continue
            
        stubs = scan_file(py_file)
        all_stubs.extend(stubs)
    
    # Print results
    print(f"Found {len(all_stubs)} stub implementations:\n")
    for stub in all_stubs:
        print(f"{stub['file']}:{stub['line']} - {stub['function']} ({stub['type']})")

if __name__ == '__main__':
    main()