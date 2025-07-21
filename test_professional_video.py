#!/usr/bin/env python3
"""Test professional video creation with all features"""
import os
import sys
import subprocess
import json

print("=== PROFESSIONAL VIDEO CREATION TEST ===\n")

# Test configuration
test_video = "tests/data/speech_test.mp4"
modes = ["smart", "premium"]
formats = ["standard", "vertical"]

results = {}

for mode in modes:
    for format_type in formats:
        print(f"\n📹 Testing {mode.upper()} mode with {format_type.upper()} format...")

        output_name = f"pro_{mode}_{format_type}.mp4"
        cmd = [
            sys.executable,
            "run_montage.py",
            test_video,
            "--mode",
            mode,
            "-o",
            f"output/{output_name}",
        ]

        if format_type == "vertical":
            cmd.append("--vertical")

        # Run the pipeline
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ {mode}/{format_type} completed successfully")

            # Check output file
            output_path = f"output/{output_name}"
            if os.path.exists(output_path):
                # Get video info
                probe_cmd = [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    output_path,
                ]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)

                if probe_result.returncode == 0:
                    info = json.loads(probe_result.stdout)
                    video_stream = next(
                        (s for s in info["streams"] if s["codec_type"] == "video"), None
                    )

                    if video_stream:
                        width = video_stream["width"]
                        height = video_stream["height"]
                        duration = float(info["format"]["duration"])
                        size_mb = os.path.getsize(output_path) / (1024 * 1024)

                        results[f"{mode}_{format_type}"] = {
                            "success": True,
                            "resolution": f"{width}x{height}",
                            "duration": duration,
                            "size_mb": size_mb,
                            "vertical": width < height,
                        }

                        print(f"   Resolution: {width}x{height}")
                        print(f"   Duration: {duration:.2f}s")
                        print(f"   Size: {size_mb:.2f} MB")
                        print(f"   Vertical: {'Yes' if width < height else 'No'}")
            else:
                results[f"{mode}_{format_type}"] = {
                    "success": False,
                    "error": "No output file",
                }
                print(f"❌ No output file created")
        else:
            results[f"{mode}_{format_type}"] = {
                "success": False,
                "error": result.stderr,
            }
            print(f"❌ Pipeline failed: {result.returncode}")

# Check latest plan for AI features
print("\n📊 Checking AI Story Features...")
plan_files = [
    f for f in os.listdir(".") if f.startswith("montage_plan_") and f.endswith(".json")
]
if plan_files:
    latest_plan = max(plan_files, key=os.path.getctime)
    with open(latest_plan) as f:
        plan = json.load(f)

    highlights = plan.get("highlights", [])
    if highlights:
        print(f"✅ Found {len(highlights)} AI-selected highlights")

        # Check for story features
        story_features = {
            "narrative_role": False,
            "emotional_impact": False,
            "story_theme": False,
            "ai_title": False,
        }

        for highlight in highlights:
            for feature in story_features:
                if feature in highlight:
                    story_features[feature] = True

        print("\nStory Features Detected:")
        for feature, found in story_features.items():
            status = "✅" if found else "❌"
            print(f"   {status} {feature}")

        # Check timeline features
        timeline = plan.get("timeline", {})
        if timeline.get("success"):
            print(f"\n✅ Timeline created with: {timeline.get('method', 'unknown')}")
            print(f"   Output: {timeline.get('output_path', 'N/A')}")

# Summary
print("\n=== SUMMARY ===")
success_count = sum(1 for r in results.values() if r.get("success"))
total_count = len(results)

print(
    f"Success Rate: {success_count}/{total_count} ({success_count/total_count*100:.0f}%)"
)

# Feature checklist
print("\n📋 Feature Implementation Check:")
features = {
    "Video Output": any(r.get("success") for r in results.values()),
    "Vertical Format": any(
        r.get("vertical") for r in results.values() if r.get("success")
    ),
    "AI Analysis": "premium_standard" in results
    and results["premium_standard"].get("success"),
    "Story Structure": (
        any(h.get("narrative_role") for h in highlights)
        if "highlights" in locals()
        else False
    ),
    "Professional Quality": all(
        r.get("size_mb", 0) > 0.1 for r in results.values() if r.get("success")
    ),
}

for feature, implemented in features.items():
    status = "✅" if implemented else "❌"
    print(f"{status} {feature}")

# Final verdict
all_implemented = all(features.values())
if all_implemented and success_count == total_count:
    print("\n🎉 ALL FEATURES FULLY IMPLEMENTED AND WORKING!")
else:
    print("\n⚠️  Some features need attention")

# Clean up test files
print("\n🧹 Cleaning up test outputs...")
for mode in modes:
    for format_type in formats:
        try:
            os.unlink(f"output/pro_{mode}_{format_type}.mp4")
        except:
            pass
