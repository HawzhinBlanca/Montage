#!/usr/bin/env python3
"""
Identify dead feature flags and unused environment variables
"""

import ast
import re
from pathlib import Path
from collections import defaultdict

def find_env_vars():
    """Find all environment variables referenced in code"""
    env_vars = defaultdict(list)
    
    patterns = [
        re.compile(r'os\.getenv\(["\']([^"\']+)["\']'),
        re.compile(r'os\.environ\[["\']([^"\']+)["\']'),
        re.compile(r'os\.environ\.get\(["\']([^"\']+)["\']'),
    ]
    
    for py_file in Path("montage").rglob("*.py"):
        try:
            with open(py_file) as f:
                content = f.read()
                
            for pattern in patterns:
                matches = pattern.findall(content)
                for match in matches:
                    env_vars[match].append(str(py_file))
        except:
            pass
    
    return env_vars

def find_feature_flags():
    """Find all feature flag usages"""
    feature_flags = defaultdict(list)
    
    # Look for feature flag patterns
    patterns = [
        re.compile(r'settings\.features\.(\w+)'),
        re.compile(r'feature_flags\.(\w+)'),
        re.compile(r'enable_(\w+)'),
        re.compile(r'USE_(\w+)'),
        re.compile(r'ENABLE_(\w+)'),
        re.compile(r'FEATURE_(\w+)'),
    ]
    
    for py_file in Path("montage").rglob("*.py"):
        try:
            with open(py_file) as f:
                content = f.read()
                
            for pattern in patterns:
                matches = pattern.findall(content)
                for match in matches:
                    feature_flags[match].append(str(py_file))
        except:
            pass
    
    return feature_flags

def analyze_feature_flag_usage():
    """Analyze which feature flags are actually used"""
    
    # Feature flags defined in settings.py
    defined_flags = {
        "enable_speaker_diarization",
        "enable_emotion_analysis", 
        "enable_smart_crop",
        "enable_audio_ducking",
        "enable_hdr_processing",
        "enable_ab_testing",
        "enable_caching",
        "cache_ttl_seconds",
        "prefer_local_models",
        "use_ollama_by_default",
        "ollama_model",
        "whisper_model_size",
    }
    
    # Find actual usages
    used_flags = set()
    usage_locations = defaultdict(list)
    
    for py_file in Path("montage").rglob("*.py"):
        try:
            with open(py_file) as f:
                content = f.read()
                
            for flag in defined_flags:
                if flag in content and str(py_file) != "montage/settings.py":
                    used_flags.add(flag)
                    usage_locations[flag].append(str(py_file))
        except:
            pass
    
    # Find dead flags
    dead_flags = defined_flags - used_flags
    
    return defined_flags, used_flags, dead_flags, usage_locations

def main():
    """Analyze and report dead feature flags"""
    
    print("🔍 Analyzing feature flags and environment variables...\n")
    
    # Analyze feature flags
    defined_flags, used_flags, dead_flags, usage_locations = analyze_feature_flag_usage()
    
    print("📊 Feature Flag Analysis:")
    print(f"   Total defined: {len(defined_flags)}")
    print(f"   Actually used: {len(used_flags)}")
    print(f"   Dead flags: {len(dead_flags)}")
    
    if dead_flags:
        print("\n❌ Dead Feature Flags (never used):")
        for flag in sorted(dead_flags):
            print(f"   - {flag}")
    
    print("\n✅ Used Feature Flags:")
    for flag in sorted(used_flags):
        locations = usage_locations[flag]
        print(f"   - {flag} ({len(locations)} locations)")
        for loc in locations[:2]:  # Show first 2 locations
            print(f"     → {loc}")
    
    # Analyze environment variables
    env_vars = find_env_vars()
    
    print(f"\n📊 Environment Variables:")
    print(f"   Total found: {len(env_vars)}")
    
    # Group by usage count
    single_use = [var for var, locs in env_vars.items() if len(locs) == 1]
    multi_use = [var for var, locs in env_vars.items() if len(locs) > 1]
    
    print(f"   Single location: {len(single_use)}")
    print(f"   Multiple locations: {len(multi_use)}")
    
    # Check for legacy/deprecated patterns
    legacy_patterns = []
    
    # USE_SETTINGS_V2 pattern
    if "USE_SETTINGS_V2" in env_vars:
        legacy_patterns.append({
            "var": "USE_SETTINGS_V2",
            "locations": env_vars["USE_SETTINGS_V2"],
            "recommendation": "Remove - settings v2 migration complete"
        })
    
    # Legacy env vars
    legacy_vars = ["USE_GPU", "MAX_WORKERS", "CACHE_TTL", "MAX_COST_USD"]
    for var in legacy_vars:
        if var in env_vars:
            legacy_patterns.append({
                "var": var,
                "locations": env_vars[var],
                "recommendation": "Migrate to structured settings"
            })
    
    if legacy_patterns:
        print("\n⚠️  Legacy Environment Variables:")
        for pattern in legacy_patterns:
            print(f"   - {pattern['var']}")
            print(f"     Recommendation: {pattern['recommendation']}")
            for loc in pattern['locations']:
                print(f"     → {loc}")
    
    # Write cleanup recommendations
    with open("feature_flag_cleanup.txt", "w") as f:
        f.write("# Feature Flag Cleanup Recommendations\n\n")
        
        if dead_flags:
            f.write("## Dead Feature Flags to Remove\n")
            for flag in sorted(dead_flags):
                f.write(f"- {flag}\n")
            f.write("\n")
        
        if legacy_patterns:
            f.write("## Legacy Environment Variables to Remove\n")
            for pattern in legacy_patterns:
                f.write(f"\n### {pattern['var']}\n")
                f.write(f"Recommendation: {pattern['recommendation']}\n")
                f.write("Locations:\n")
                for loc in pattern['locations']:
                    f.write(f"- {loc}\n")
    
    print(f"\n📄 Cleanup recommendations saved to: feature_flag_cleanup.txt")

if __name__ == "__main__":
    main()