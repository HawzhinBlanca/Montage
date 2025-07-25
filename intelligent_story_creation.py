#!/usr/bin/env python3
"""
INTELLIGENT STORY CREATION - Creative Director Mode
Extract the most compelling narrative from David Sinclair interview
Focus on emotional impact, dramatic tension, and viewer engagement
"""

import json
import re
from typing import List, Dict, Tuple


def analyze_transcript_for_story_beats(analysis: Dict) -> Dict:
    """Extract story beats with dramatic potential"""

    words = analysis.get("words", [])
    transcript = analysis.get("transcript", "")

    # Story beats identified by creative director analysis
    story_beats = {
        "authority_establishment": {
            "keywords": [
                "professor",
                "david",
                "sinclair",
                "harvard",
                "renowned",
                "prominent",
                "scientist",
            ],
            "emotional_weight": 8,
            "purpose": "Establish credibility - why should we listen?",
        },
        "revolutionary_claim": {
            "keywords": [
                "aging",
                "disease",
                "address",
                "hundreds",
                "thousands",
                "die",
                "every",
                "day",
            ],
            "emotional_weight": 9,
            "purpose": "The shocking revelation that changes everything",
        },
        "personal_transformation": {
            "keywords": [
                "used",
                "older",
                "birthdays",
                "wrinkles",
                "went",
                "away",
                "serena",
                "changed",
            ],
            "emotional_weight": 8,
            "purpose": "Proof it works - visual transformation",
        },
        "father_proof": {
            "keywords": [
                "father",
                "85",
                "disease",
                "ailment",
                "best",
                "time",
                "life",
                "traveling",
            ],
            "emotional_weight": 9,
            "purpose": "Living example of successful aging",
        },
        "adversity_revelation": {
            "keywords": [
                "adversity",
                "mode",
                "abundance",
                "comfort",
                "genes",
                "turn",
                "off",
                "survival",
            ],
            "emotional_weight": 8,
            "purpose": "The counterintuitive truth about health",
        },
        "concrete_solutions": {
            "keywords": [
                "skip",
                "breakfast",
                "plants",
                "exercise",
                "sauna",
                "temperature",
                "fasting",
            ],
            "emotional_weight": 7,
            "purpose": "Actionable steps viewers can take today",
        },
        "future_breakthrough": {
            "keywords": [
                "pill",
                "reverse",
                "aging",
                "reset",
                "decade",
                "human",
                "trials",
                "next",
                "year",
            ],
            "emotional_weight": 10,
            "purpose": "The incredible future that's almost here",
        },
        "economic_impact": {
            "keywords": [
                "38",
                "trillion",
                "economic",
                "benefits",
                "productivity",
                "health",
                "system",
            ],
            "emotional_weight": 6,
            "purpose": "Societal implications - this affects everyone",
        },
    }

    # Find segments for each story beat
    story_segments = []

    for i in range(len(words) - 20):  # Look at windows of 20+ words
        window_words = words[i : i + 20]
        window_text = " ".join([w["word"].lower().strip() for w in window_words])

        # Score this window against each story beat
        for beat_name, beat_info in story_beats.items():
            score = 0
            matches = []

            for keyword in beat_info["keywords"]:
                if keyword.lower() in window_text:
                    score += 1
                    matches.append(keyword)

            # If we have good keyword matches, extend the segment
            if score >= 3:
                # Extend backward and forward to get complete sentences
                start_idx = i
                end_idx = i + 20

                # Extend backward to sentence start
                while start_idx > 0:
                    prev_word = words[start_idx - 1]["word"]
                    if prev_word.strip().endswith((".", "!", "?")) or start_idx == 0:
                        break
                    start_idx -= 1

                # Extend forward to sentence end
                while end_idx < len(words) - 1:
                    curr_word = words[end_idx]["word"]
                    if curr_word.strip().endswith((".", "!", "?")):
                        end_idx += 1
                        break
                    end_idx += 1

                # Create segment
                segment_words = words[start_idx:end_idx]
                if len(segment_words) > 5:  # Minimum segment length
                    segment = {
                        "beat_type": beat_name,
                        "text": " ".join([w["word"] for w in segment_words]).strip(),
                        "start_time": segment_words[0]["start"],
                        "end_time": segment_words[-1]["end"],
                        "duration": segment_words[-1]["end"]
                        - segment_words[0]["start"],
                        "keyword_matches": matches,
                        "score": score * beat_info["emotional_weight"],
                        "emotional_weight": beat_info["emotional_weight"],
                        "purpose": beat_info["purpose"],
                        "speaker": segment_words[0].get("speaker", "unknown"),
                    }

                    # Only add if it's a reasonable duration and not duplicate
                    if 5 <= segment["duration"] <= 45:
                        # Check for duplicates
                        is_duplicate = False
                        for existing in story_segments:
                            # If segments overlap significantly, keep the higher scoring one
                            overlap_start = max(
                                segment["start_time"], existing["start_time"]
                            )
                            overlap_end = min(segment["end_time"], existing["end_time"])
                            overlap_duration = max(0, overlap_end - overlap_start)

                            if (
                                overlap_duration
                                > min(segment["duration"], existing["duration"]) * 0.5
                            ):
                                if segment["score"] > existing["score"]:
                                    story_segments.remove(existing)
                                else:
                                    is_duplicate = True
                                break

                        if not is_duplicate:
                            story_segments.append(segment)

    # Sort by score and select best segments for each beat type
    story_segments.sort(key=lambda x: x["score"], reverse=True)

    # Select top segments ensuring beat diversity
    selected_beats = set()
    final_segments = []

    for segment in story_segments:
        if len(final_segments) >= 8:  # Max 8 segments for ~2 minute video
            break

        beat_type = segment["beat_type"]

        # Prioritize getting at least one of each important beat type
        important_beats = [
            "authority_establishment",
            "revolutionary_claim",
            "personal_transformation",
            "future_breakthrough",
            "adversity_revelation",
        ]

        if beat_type in important_beats and beat_type not in selected_beats:
            final_segments.append(segment)
            selected_beats.add(beat_type)
        elif beat_type not in important_beats and len(final_segments) < 8:
            final_segments.append(segment)
            selected_beats.add(beat_type)

    # Sort final segments by time order for coherent flow
    final_segments.sort(key=lambda x: x["start_time"])

    return {
        "segments": final_segments,
        "total_duration": sum(s["duration"] for s in final_segments),
        "story_flow": [s["beat_type"] for s in final_segments],
        "emotional_arc": [s["emotional_weight"] for s in final_segments],
    }


def analyze_story_quality(story_structure: Dict) -> Dict:
    """Analyze the story quality and provide creative direction"""

    segments = story_structure["segments"]

    if not segments:
        return {
            "rating": 0,
            "issues": ["No compelling segments found"],
            "recommendations": ["Content may not be suitable for story-driven video"],
        }

    # Story quality metrics
    beat_diversity = len(set(s["beat_type"] for s in segments))
    emotional_range = max(s["emotional_weight"] for s in segments) - min(
        s["emotional_weight"] for s in segments
    )
    duration_balance = story_structure["total_duration"] / len(
        segments
    )  # Average segment length

    # Check for key story elements
    has_hook = any(
        s["beat_type"] in ["authority_establishment", "revolutionary_claim"]
        for s in segments
    )
    has_transformation = any(
        s["beat_type"] in ["personal_transformation", "father_proof"] for s in segments
    )
    has_solution = any(
        s["beat_type"] in ["concrete_solutions", "adversity_revelation"]
        for s in segments
    )
    has_future = any(s["beat_type"] == "future_breakthrough" for s in segments)

    # Calculate story quality score
    score = 0
    issues = []
    recommendations = []

    # Beat diversity (max 3 points)
    if beat_diversity >= 5:
        score += 3
    elif beat_diversity >= 3:
        score += 2
    else:
        score += 1
        issues.append(f"Limited story diversity ({beat_diversity} beat types)")
        recommendations.append(
            "Need more varied story elements for compelling narrative"
        )

    # Emotional arc (max 2 points)
    if emotional_range >= 3:
        score += 2
    elif emotional_range >= 2:
        score += 1
    else:
        issues.append("Flat emotional arc")
        recommendations.append("Need higher emotional peaks and valleys")

    # Essential story elements (max 4 points)
    if has_hook:
        score += 1
    else:
        issues.append("Missing strong opening hook")
        recommendations.append("Need attention-grabbing opener")

    if has_transformation:
        score += 1
    else:
        issues.append("Missing personal transformation story")
        recommendations.append("Need proof that the solution works")

    if has_solution:
        score += 1
    else:
        issues.append("Missing actionable solutions")
        recommendations.append("Need concrete steps viewers can take")

    if has_future:
        score += 1
    else:
        issues.append("Missing future vision")
        recommendations.append("Need hope/excitement about what's coming")

    # Duration balance (max 1 point)
    if 8 <= duration_balance <= 20:  # 8-20 second segments are ideal
        score += 1
    else:
        issues.append(f"Poor segment pacing (avg {duration_balance:.1f}s per segment)")
        recommendations.append("Adjust segment lengths for better flow")

    # Overall rating out of 10
    rating = score

    # Quality assessment
    if rating >= 9:
        quality = "EXCEPTIONAL - Ready for professional production"
    elif rating >= 7:
        quality = "STRONG - Minor refinements needed"
    elif rating >= 5:
        quality = "DECENT - Needs significant improvement"
    elif rating >= 3:
        quality = "WEAK - Major story issues"
    else:
        quality = "POOR - Requires complete restructure"

    return {
        "rating": rating,
        "quality": quality,
        "beat_diversity": beat_diversity,
        "emotional_range": emotional_range,
        "duration_balance": duration_balance,
        "has_essential_elements": {
            "hook": has_hook,
            "transformation": has_transformation,
            "solution": has_solution,
            "future": has_future,
        },
        "issues": issues,
        "recommendations": recommendations,
    }


def main():
    """Creative Director: Analyze story potential and create compelling structure"""

    print("🎭 INTELLIGENT STORY CREATION - Creative Director Mode")
    print("=" * 70)

    # Load the full video analysis
    try:
        with open("/Users/hawzhin/Montage/full_video_analysis.json", "r") as f:
            analysis = json.load(f)
        print(f"✅ Loaded analysis: {len(analysis.get('words', []))} words")
    except FileNotFoundError:
        print("❌ No analysis file found - run video analysis first")
        return

    # Extract story structure
    print("\n🔍 EXTRACTING STORY BEATS...")
    story_structure = analyze_transcript_for_story_beats(analysis)

    segments = story_structure["segments"]
    print(f"   Found {len(segments)} compelling segments")
    print(f"   Total duration: {story_structure['total_duration']:.1f} seconds")
    print(f"   Story flow: {' → '.join(story_structure['story_flow'])}")

    # Analyze story quality
    print("\n🎯 STORY QUALITY ANALYSIS...")
    quality_analysis = analyze_story_quality(story_structure)

    print(f"   📊 Rating: {quality_analysis['rating']}/10")
    print(f"   🏆 Quality: {quality_analysis['quality']}")
    print(f"   🎭 Beat diversity: {quality_analysis['beat_diversity']} types")
    print(f"   💓 Emotional range: {quality_analysis['emotional_range']} points")
    print(f"   ⏱️  Average segment: {quality_analysis['duration_balance']:.1f}s")

    # Show essential elements
    elements = quality_analysis["has_essential_elements"]
    print(f"\n🎬 ESSENTIAL STORY ELEMENTS:")
    print(f"   {'✅' if elements['hook'] else '❌'} Strong Hook/Opening")
    print(f"   {'✅' if elements['transformation'] else '❌'} Personal Transformation")
    print(f"   {'✅' if elements['solution'] else '❌'} Actionable Solutions")
    print(f"   {'✅' if elements['future'] else '❌'} Future Vision")

    # Show detailed segments
    print(f"\n📚 STORY SEGMENTS:")
    for i, segment in enumerate(segments, 1):
        print(f"\n   {i}. {segment['beat_type'].upper().replace('_', ' ')}")
        print(
            f"      ⏰ {segment['start_time']:.1f}s - {segment['end_time']:.1f}s ({segment['duration']:.1f}s)"
        )
        print(
            f"      💯 Score: {segment['score']} | Emotion: {segment['emotional_weight']}/10"
        )
        print(f"      🎯 Purpose: {segment['purpose']}")
        print(f"      💬 Text: \"{segment['text'][:100]}...\"")
        print(f"      🔍 Keywords: {', '.join(segment['keyword_matches'])}")

    # Show issues and recommendations
    if quality_analysis["issues"]:
        print(f"\n⚠️  CREATIVE ISSUES:")
        for issue in quality_analysis["issues"]:
            print(f"   • {issue}")

    if quality_analysis["recommendations"]:
        print(f"\n🎨 CREATIVE RECOMMENDATIONS:")
        for rec in quality_analysis["recommendations"]:
            print(f"   • {rec}")

    # Creative Director Decision
    print(f"\n" + "=" * 70)
    print(f"🎬 CREATIVE DIRECTOR VERDICT:")
    print(f"=" * 70)

    if quality_analysis["rating"] >= 7:
        print(f"✅ APPROVED FOR PRODUCTION")
        print(f"   This story structure has strong narrative potential.")
        print(f"   The segments create a compelling emotional journey.")
        print(f"   Ready to proceed with video creation.")

        # Save the approved story structure
        output = {
            "story_structure": story_structure,
            "quality_analysis": quality_analysis,
            "creative_director_approval": True,
            "approved_segments": segments,
        }

        with open("approved_story_structure.json", "w") as f:
            json.dump(output, f, indent=2)

        print(f"   💾 Saved approved structure to approved_story_structure.json")

    elif quality_analysis["rating"] >= 5:
        print(f"⚠️  CONDITIONAL APPROVAL - NEEDS REFINEMENT")
        print(f"   Story has potential but requires improvements.")
        print(f"   Address the issues above before proceeding.")
        print(f"   Consider re-editing segments for better flow.")

    else:
        print(f"❌ REJECTED - STORY NEEDS MAJOR WORK")
        print(f"   Current structure lacks compelling narrative.")
        print(f"   Content may not be suitable for story-driven format.")
        print(f"   Consider different content or approach.")

    print(f"\n🎭 Creative Director Analysis Complete")


if __name__ == "__main__":
    main()
