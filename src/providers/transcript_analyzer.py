"""Advanced LLM-based transcript analysis with caching"""

import json
import logging
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import Counter
import openai
import av

from config import Config
from db import Database
from metrics import metrics
from budget_guard import priced, BudgetExceededError

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """Represents a segment of transcript with metadata"""

    text: str
    start_time: float
    end_time: float
    speaker: Optional[str] = None
    confidence: float = 1.0

    @property
    def duration(self):
        return self.end_time - self.start_time


@dataclass
class AnalysisResult:
    """Result of transcript analysis"""

    score: float
    tf_idf_score: float
    audio_rms: float
    visual_energy: float
    start_time: float
    end_time: float
    text: str
    reason: str
    keywords: List[str]


class TranscriptAnalyzer:
    """Analyzes transcripts using LLM with chunk-based processing and caching"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.OPENAI_API_KEY
        openai.api_key = self.api_key
        self.db = Database()
        self.chunk_size = 2000  # tokens per chunk
        self.overlap_size = 200  # overlap between chunks

    def analyze_transcript(
        self,
        job_id: str,
        transcript_segments: List[TranscriptSegment],
        audio_path: str,
        video_path: str,
    ) -> List[AnalysisResult]:
        """
        Analyze transcript with multi-modal scoring.

        Score = TF-IDF × Audio_RMS × Visual_Energy
        """
        # Check cache first
        file_hash = self._calculate_file_hash(audio_path)
        cached_result = self._get_cached_analysis(file_hash)

        if cached_result:
            logger.info(f"Using cached transcript analysis for {file_hash}")
            return cached_result

        logger.info(f"Analyzing transcript with {len(transcript_segments)} segments")

        # Extract features
        audio_features = self._extract_audio_features(audio_path, transcript_segments)
        visual_features = self._extract_visual_features(video_path, transcript_segments)

        # Chunk transcript for LLM analysis
        chunks = self._chunk_transcript(transcript_segments)
        logger.info(f"Split transcript into {len(chunks)} chunks")

        # Analyze each chunk
        all_results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Analyzing chunk {i+1}/{len(chunks)}")
            chunk_results = self._analyze_chunk(
                chunk, transcript_segments, audio_features, visual_features
            )
            all_results.extend(chunk_results)

        # Merge and rank results
        final_results = self._merge_and_rank_results(all_results)

        # Cache results
        self._cache_analysis(file_hash, final_results, "openai")

        # Track metrics
        for result in final_results:
            metrics.track_highlight_score(result.score)

        return final_results

    def _chunk_transcript(self, segments: List[TranscriptSegment]) -> List[str]:
        """Split transcript into overlapping chunks for LLM processing"""
        full_text = " ".join([s.text for s in segments])
        words = full_text.split()

        chunks = []
        chunk_words = []
        word_count = 0

        for word in words:
            chunk_words.append(word)
            word_count += 1

            # Rough token estimation (1 word ≈ 1.3 tokens)
            if word_count * 1.3 >= self.chunk_size:
                chunks.append(" ".join(chunk_words))

                # Keep overlap for context
                overlap_start = max(0, len(chunk_words) - int(self.overlap_size / 1.3))
                chunk_words = chunk_words[overlap_start:]
                word_count = len(chunk_words)

        # Add remaining words
        if chunk_words:
            chunks.append(" ".join(chunk_words))

        return chunks

    @priced(
        "openai",
        lambda r: 0.02
        * len(r.get("choices", [{}])[0].get("message", {}).get("content", ""))
        / 1000,
    )
    def _analyze_chunk(
        self,
        chunk_text: str,
        all_segments: List[TranscriptSegment],
        audio_features: Dict,
        visual_features: Dict,
    ) -> List[AnalysisResult]:
        """Analyze a chunk of transcript using LLM"""

        prompt = f"""Analyze this transcript excerpt and identify the most interesting moments for video highlights.
Look for:
- Key insights or revelations
- Emotional moments
- Important announcements
- Compelling stories
- Turning points in conversation

Transcript:
{chunk_text}

Return a JSON array of highlights with:
- text: The exact quote
- reason: Why this is interesting
- keywords: Key terms that make this important

Focus on complete thoughts (15-60 seconds when spoken).
"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a video editor finding the best moments in content.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            content = response.choices[0].message.content

            # Parse JSON response
            try:
                highlights_data = json.loads(content)
                if not isinstance(highlights_data, list):
                    highlights_data = [highlights_data]
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON")
                highlights_data = []

            # Convert to AnalysisResult objects
            results = []
            for highlight in highlights_data:
                if not isinstance(highlight, dict):
                    continue

                text = highlight.get("text", "")
                if not text:
                    continue

                # Find matching segment
                segment = self._find_segment_for_text(text, all_segments)
                if not segment:
                    continue

                # Calculate TF-IDF score
                tf_idf_score = self._calculate_tfidf_score(
                    text, highlight.get("keywords", [])
                )

                # Get audio/visual features for this segment
                audio_rms = audio_features.get(segment.start_time, 0.5)
                visual_energy = visual_features.get(segment.start_time, 0.5)

                # Calculate final score
                score = tf_idf_score * audio_rms * visual_energy

                result = AnalysisResult(
                    score=score,
                    tf_idf_score=tf_idf_score,
                    audio_rms=audio_rms,
                    visual_energy=visual_energy,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    text=text,
                    reason=highlight.get("reason", ""),
                    keywords=highlight.get("keywords", []),
                )

                results.append(result)

            return results

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            if "budget" in str(e).lower():
                raise BudgetExceededError(str(e))
            return []

    def _find_segment_for_text(
        self, text: str, segments: List[TranscriptSegment]
    ) -> Optional[TranscriptSegment]:
        """Find the transcript segment containing the given text"""
        text_lower = text.lower().strip()

        for segment in segments:
            if text_lower in segment.text.lower():
                return segment

        # Fuzzy match - find segment with most word overlap
        text_words = set(text_lower.split())
        best_match = None
        best_overlap = 0

        for segment in segments:
            segment_words = set(segment.text.lower().split())
            overlap = len(text_words & segment_words)

            if overlap > best_overlap and overlap >= len(text_words) * 0.5:
                best_overlap = overlap
                best_match = segment

        return best_match

    def _calculate_tfidf_score(self, text: str, keywords: List[str]) -> float:
        """Calculate TF-IDF score for text importance"""
        # Simple TF-IDF approximation
        words = text.lower().split()
        word_freq = Counter(words)

        # Boost score for keywords
        keyword_boost = 0
        for keyword in keywords:
            if keyword.lower() in word_freq:
                keyword_boost += 0.2

        # Calculate term frequency
        max_freq = max(word_freq.values()) if word_freq else 1
        avg_tf = sum(freq / max_freq for freq in word_freq.values()) / len(word_freq)

        # Simple IDF based on common words
        common_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
        }
        unique_words = [w for w in words if w not in common_words]
        idf_score = len(unique_words) / len(words) if words else 0

        # Combine scores
        tf_idf = (avg_tf * idf_score) + keyword_boost

        # Normalize to 0-1 range
        return min(1.0, tf_idf)

    def _extract_audio_features(
        self, audio_path: str, segments: List[TranscriptSegment]
    ) -> Dict[float, float]:
        """Extract audio RMS energy for each segment"""
        features = {}

        try:
            container = av.open(audio_path)
            audio_stream = next(s for s in container.streams if s.type == "audio")

            # Simple RMS calculation for each segment
            for segment in segments:
                # Seek to segment start
                container.seek(int(segment.start_time * 1000000))

                rms_values = []
                for frame in container.decode(audio_stream):
                    if frame.time > segment.end_time:
                        break

                    # Calculate RMS
                    samples = frame.to_ndarray()
                    if samples.size > 0:
                        rms = np.sqrt(np.mean(samples**2))
                        rms_values.append(rms)

                # Average RMS for segment, normalized to 0-1
                avg_rms = np.mean(rms_values) if rms_values else 0.5
                features[segment.start_time] = min(1.0, avg_rms * 10)  # Scale and cap

            container.close()

        except Exception as e:
            logger.error(f"Audio feature extraction failed: {e}")
            # Return default values
            return {s.start_time: 0.5 for s in segments}

        return features

    def _extract_visual_features(
        self, video_path: str, segments: List[TranscriptSegment]
    ) -> Dict[float, float]:
        """Extract visual energy (motion) for each segment"""
        features = {}

        try:
            container = av.open(video_path)
            video_stream = next(s for s in container.streams if s.type == "video")

            for segment in segments:
                # Sample a few frames from the segment
                container.seek(int(segment.start_time * 1000000))

                prev_frame = None
                motion_scores = []

                frame_count = 0
                for frame in container.decode(video_stream):
                    if frame.time > segment.end_time:
                        break

                    # Sample every 10th frame
                    frame_count += 1
                    if frame_count % 10 != 0:
                        continue

                    # Convert to grayscale numpy array
                    img = frame.to_ndarray(format="gray")

                    if prev_frame is not None:
                        # Calculate motion as frame difference
                        diff = np.abs(img.astype(float) - prev_frame.astype(float))
                        motion_score = np.mean(diff) / 255.0
                        motion_scores.append(motion_score)

                    prev_frame = img

                # Average motion score, boosted and capped
                avg_motion = np.mean(motion_scores) if motion_scores else 0.5
                features[segment.start_time] = min(1.0, avg_motion * 5)

            container.close()

        except Exception as e:
            logger.error(f"Visual feature extraction failed: {e}")
            # Return default values
            return {s.start_time: 0.5 for s in segments}

        return features

    def _merge_and_rank_results(
        self, results: List[AnalysisResult]
    ) -> List[AnalysisResult]:
        """Merge overlapping results and rank by score"""
        # Remove duplicates and merge overlapping segments
        merged = []

        # Sort by start time
        results.sort(key=lambda r: r.start_time)

        for result in results:
            # Check if this overlaps with any existing result
            overlap_found = False

            for i, existing in enumerate(merged):
                # Check for time overlap
                if (
                    result.start_time <= existing.end_time
                    and result.end_time >= existing.start_time
                ):

                    # Keep the higher scoring one
                    if result.score > existing.score:
                        merged[i] = result

                    overlap_found = True
                    break

            if not overlap_found:
                merged.append(result)

        # Sort by score (highest first)
        merged.sort(key=lambda r: r.score, reverse=True)

        # Keep top results
        max_results = 10
        return merged[:max_results]

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of file for caching"""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def _get_cached_analysis(self, file_hash: str) -> Optional[List[AnalysisResult]]:
        """Retrieve cached analysis results"""
        cache_record = self.db.find_one("transcript_cache", {"src_hash": file_hash})

        if cache_record:
            try:
                data = json.loads(cache_record["transcript_data"])

                # Convert back to AnalysisResult objects
                results = []
                for item in data:
                    result = AnalysisResult(
                        score=item["score"],
                        tf_idf_score=item["tf_idf_score"],
                        audio_rms=item["audio_rms"],
                        visual_energy=item["visual_energy"],
                        start_time=item["start_time"],
                        end_time=item["end_time"],
                        text=item["text"],
                        reason=item["reason"],
                        keywords=item["keywords"],
                    )
                    results.append(result)

                return results

            except Exception as e:
                logger.error(f"Failed to parse cached analysis: {e}")

        return None

    def _cache_analysis(
        self, file_hash: str, results: List[AnalysisResult], provider: str
    ):
        """Cache analysis results"""
        # Convert to JSON-serializable format
        data = []
        for result in results:
            data.append(
                {
                    "score": result.score,
                    "tf_idf_score": result.tf_idf_score,
                    "audio_rms": result.audio_rms,
                    "visual_energy": result.visual_energy,
                    "start_time": result.start_time,
                    "end_time": result.end_time,
                    "text": result.text,
                    "reason": result.reason,
                    "keywords": result.keywords,
                }
            )

        # Store in database
        self.db.insert(
            "transcript_cache",
            {
                "src_hash": file_hash,
                "transcript_data": json.dumps(data),
                "provider": provider,
            },
        )

        logger.info(f"Cached analysis for {file_hash}")


# Example ASR integration
@priced("whisper", lambda r: 0.006 * r.get("duration", 0) / 60)
def transcribe_audio_whisper(audio_path: str, job_id: str) -> List[TranscriptSegment]:
    """Transcribe audio using OpenAI Whisper API"""

    with open(audio_path, "rb") as audio_file:
        response = openai.Audio.transcribe(
            model="whisper-1", file=audio_file, response_format="verbose_json"
        )

    segments = []
    for segment in response.get("segments", []):
        segments.append(
            TranscriptSegment(
                text=segment["text"],
                start_time=segment["start"],
                end_time=segment["end"],
                confidence=segment.get("confidence", 1.0),
            )
        )

    # Track cost
    duration = response.get("duration", 0)

    return segments


# Integration with SmartVideoEditor
def analyze_video_content(
    job_id: str, video_path: str, audio_path: str
) -> List[Dict[str, Any]]:
    """Main entry point for video content analysis"""

    analyzer = TranscriptAnalyzer()

    # Step 1: Transcribe audio
    logger.info("Transcribing audio...")
    transcript_segments = transcribe_audio_whisper(audio_path, job_id)

    # Step 2: Analyze transcript with multi-modal features
    logger.info("Analyzing content...")
    results = analyzer.analyze_transcript(
        job_id, transcript_segments, audio_path, video_path
    )

    # Convert to highlight format
    highlights = []
    for result in results:
        highlights.append(
            {
                "start_time": result.start_time,
                "end_time": result.end_time,
                "score": result.score,
                "text": result.text,
                "reason": result.reason,
                "tf_idf_score": result.tf_idf_score,
                "audio_rms": result.audio_rms,
                "visual_energy": result.visual_energy,
            }
        )

    # Store in database
    db = Database()
    for highlight in highlights:
        db.insert(
            "highlight",
            {
                "job_id": job_id,
                "start_time": highlight["start_time"],
                "end_time": highlight["end_time"],
                "score": highlight["score"],
                "tf_idf_score": highlight["tf_idf_score"],
                "audio_rms": highlight["audio_rms"],
                "visual_energy": highlight["visual_energy"],
                "transcript": highlight["text"],
            },
        )

    return highlights


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    analyzer = TranscriptAnalyzer()

    # Mock transcript
    segments = [
        TranscriptSegment("This is a major breakthrough in AI research.", 0, 5),
        TranscriptSegment("We discovered something incredible today.", 10, 15),
        TranscriptSegment("The results exceeded all our expectations.", 20, 25),
    ]

    # Analyze (would need real audio/video files)
    # results = analyzer.analyze_transcript("job-001", segments, "audio.mp3", "video.mp4")
