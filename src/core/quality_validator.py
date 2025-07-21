#!/usr/bin/env python3
"""
Production-ready quality validation for Montage pipeline
Hard fails on quality issues to prevent bad output delivery
"""
import os
import re
from typing import Dict, List, Any, Tuple


class QualityValidationError(Exception):
    """Raised when quality validation fails"""

    pass


def validate_subtitle_quality(subtitle_files: List[str]) -> List[str]:
    """Validate subtitle files for timing overlaps and formatting issues"""
    issues = []

    for subtitle_file in subtitle_files:
        if not os.path.exists(subtitle_file):
            issues.append(f"Missing subtitle file: {subtitle_file}")
            continue

        try:
            with open(subtitle_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse SRT timing
            timing_pattern = (
                r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})"
            )
            timings = []

            for match in re.finditer(timing_pattern, content):
                start_h, start_m, start_s, start_ms = map(int, match.groups()[:4])
                end_h, end_m, end_s, end_ms = map(int, match.groups()[4:])

                start_total_ms = (
                    start_h * 3600 + start_m * 60 + start_s
                ) * 1000 + start_ms
                end_total_ms = (end_h * 3600 + end_m * 60 + end_s) * 1000 + end_ms

                timings.append((start_total_ms, end_total_ms))

            # Check for timing issues
            for i, (start, end) in enumerate(timings):
                # Check for invalid timing (start >= end)
                if start >= end:
                    issues.append(
                        f"Invalid timing in {subtitle_file} line {i+1}: start >= end"
                    )

                # Check for overlaps with next subtitle
                if i < len(timings) - 1:
                    next_start = timings[i + 1][0]
                    if end > next_start:
                        issues.append(
                            f"Timing overlap in {subtitle_file} between lines {i+1} and {i+2}"
                        )

        except Exception as e:
            issues.append(f"Error reading subtitle file {subtitle_file}: {e}")

    return issues


def validate_highlight_quality(highlights: List[Dict]) -> List[str]:
    """Validate highlight selection quality"""
    issues = []

    if not highlights:
        issues.append("No highlights generated")
        return issues

    # Check minimum highlight count
    if len(highlights) < 2:
        issues.append(f"Insufficient highlights: {len(highlights)} < 2 minimum")

    # Check for duplicate text detection
    for i, highlight in enumerate(highlights):
        text = highlight.get("text", "")
        if not text:
            issues.append(f"Highlight {i+1} has empty text")
            continue

        # Check for consecutive duplicate words
        words = text.split()
        for j in range(len(words) - 1):
            if words[j].lower().strip(".,!?;:") == words[j + 1].lower().strip(".,!?;:"):
                issues.append(
                    f"Highlight {i+1} contains duplicate words: '{words[j]} {words[j+1]}'"
                )
                break

    # Check audio energy diversity
    energy_values = [
        h.get("audio_energy") for h in highlights if h.get("audio_energy") is not None
    ]
    if energy_values:
        unique_values = len(
            set(f"{v:.3f}" for v in energy_values)
        )  # Round to 3 decimals
        if unique_values == 1:
            issues.append(
                f"All highlights have identical audio energy: {energy_values[0]}"
            )
        elif unique_values < len(energy_values) * 0.7:  # Less than 70% unique
            issues.append(
                f"Low audio energy diversity: {unique_values}/{len(energy_values)} unique values"
            )

    # Check timing validity
    for i, highlight in enumerate(highlights):
        start_ms = highlight.get("start_ms", 0)
        end_ms = highlight.get("end_ms", 0)

        if start_ms >= end_ms:
            issues.append(
                f"Highlight {i+1} has invalid timing: start {start_ms} >= end {end_ms}"
            )

        duration = (end_ms - start_ms) / 1000.0
        if duration < 3.0:
            issues.append(f"Highlight {i+1} too short: {duration:.1f}s < 3.0s minimum")
        elif duration > 30.0:
            issues.append(f"Highlight {i+1} too long: {duration:.1f}s > 30.0s maximum")

    return issues


def validate_transcription_quality(analysis: Dict) -> List[str]:
    """Validate transcription quality metrics"""
    issues = []

    word_count = analysis.get("word_count", 0)
    speaker_turns = analysis.get("speaker_turns", 0)

    # Check minimum transcription
    if word_count < 100:
        issues.append(f"Insufficient transcription: {word_count} words < 100 minimum")

    # Check speaker turn sanity
    if speaker_turns > 200:
        issues.append(
            f"Excessive speaker turns: {speaker_turns} > 200 (over-segmentation)"
        )
    elif speaker_turns < 1:
        issues.append(f"No speaker turns detected: {speaker_turns}")

    return issues


def validate_output_files(output_video: str, subtitle_files: List[str]) -> List[str]:
    """Validate output file integrity"""
    issues = []

    # Check video output
    if not output_video or not os.path.exists(output_video):
        issues.append(f"Missing output video: {output_video}")
    else:
        # Check video file size
        size = os.path.getsize(output_video)
        if size < 1024 * 1024:  # Less than 1MB
            issues.append(f"Output video suspiciously small: {size} bytes")

    # Check subtitle files
    for subtitle_file in subtitle_files:
        if not os.path.exists(subtitle_file):
            issues.append(f"Missing subtitle file: {subtitle_file}")
        else:
            size = os.path.getsize(subtitle_file)
            if size < 50:  # Less than 50 bytes
                issues.append(
                    f"Subtitle file suspiciously small: {subtitle_file} ({size} bytes)"
                )

    return issues


def validate_pipeline_output(
    highlights: List[Dict],
    analysis: Dict,
    timeline: Dict,
    output_video: str,
    subtitle_files: List[str],
) -> Tuple[bool, List[str]]:
    """
    Comprehensive quality validation for pipeline output
    Returns (is_valid, issues_list)
    Raises QualityValidationError on critical failures
    """
    all_issues = []

    # Validate each component
    all_issues.extend(validate_highlight_quality(highlights))
    all_issues.extend(validate_transcription_quality(analysis))
    all_issues.extend(validate_subtitle_quality(subtitle_files))
    all_issues.extend(validate_output_files(output_video, subtitle_files))

    # Check timeline success
    if not timeline.get("success", False):
        all_issues.append("Timeline generation failed")

    # Classify issues by severity
    critical_patterns = [
        "No highlights generated",
        "Timeline generation failed",
        "Missing output video",
        "Invalid timing",
        "Timing overlap",
    ]

    critical_issues = [
        issue
        for issue in all_issues
        if any(pattern in issue for pattern in critical_patterns)
    ]

    # Hard fail on critical issues
    if critical_issues:
        raise QualityValidationError(
            f"Critical quality failures:\n"
            + "\n".join(f"- {issue}" for issue in critical_issues)
        )

    # Return validation result
    is_valid = len(all_issues) == 0
    return is_valid, all_issues


# Production quality gate
def enforce_production_quality(pipeline_result: Dict) -> Dict:
    """
    Enforce production quality standards
    Modifies pipeline_result with quality metrics
    Raises QualityValidationError on failures
    """
    highlights = pipeline_result.get("highlights", [])
    analysis = pipeline_result.get("analysis", {})
    timeline = pipeline_result.get("timeline", {})
    output_video = pipeline_result.get("output_video", "")
    subtitle_files = []  # Will be populated from pipeline result if available

    try:
        is_valid, issues = validate_pipeline_output(
            highlights, analysis, timeline, output_video, subtitle_files
        )

        # Add quality metrics to result
        pipeline_result["quality"] = {
            "is_valid": is_valid,
            "issues": issues,
            "validation_passed": True,
            "critical_failures": 0,
        }

        if not is_valid:
            print(f"⚠️  Quality issues detected ({len(issues)} total):")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ Quality validation passed - output meets production standards")

        return pipeline_result

    except QualityValidationError as e:
        # Hard failure - do not deliver output
        pipeline_result["quality"] = {
            "is_valid": False,
            "issues": [str(e)],
            "validation_passed": False,
            "critical_failures": 1,
        }

        print(f"❌ QUALITY VALIDATION FAILED - Output rejected")
        print(f"   {e}")

        # Re-raise to stop pipeline
        raise e
