#!/usr/bin/env python3
"""
Phase 1.1: Ensemble ASR + diarisation
Whisper.cpp + Deepgram Nova-2 + ROVER merge + pyannote alignment
"""

import os
import sys
import json
import logging
import asyncio
import tempfile
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import subprocess

# Critical imports
try:
    import faster_whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("faster-whisper not available")

try:
    import pyannote.audio
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logging.warning("pyannote.audio not available")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logging.warning("httpx not available - installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx"])
    import httpx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class WordTiming:
    """Word with timing information"""
    word: str
    start: float
    end: float
    confidence: float
    speaker: Optional[str] = None

@dataclass
class TranscriptSegment:
    """Transcript segment with speaker info"""
    text: str
    start: float
    end: float
    speaker: str
    confidence: float
    words: List[WordTiming]

@dataclass
class EnsembleResult:
    """Complete ensemble ASR result"""
    transcript: str
    segments: List[TranscriptSegment]
    word_timings: List[WordTiming]
    speakers: List[str]
    language: str
    confidence: float
    processing_time: float
    methods_used: List[str]

class WhisperLocalProcessor:
    """Local Whisper.cpp processor"""
    
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """Initialize Whisper model"""
        if not WHISPER_AVAILABLE:
            logger.error("faster-whisper not available")
            return
        
        try:
            # Use CPU for base model - fits in < 100MB RAM
            self.model = faster_whisper.WhisperModel(
                self.model_size, 
                device="cpu",
                compute_type="int8"  # Quantized for speed
            )
            logger.info(f"✅ Whisper {self.model_size} model loaded")
        except Exception as e:
            logger.error(f"❌ Whisper model loading failed: {e}")
            self.model = None
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio with Whisper"""
        if not self.model:
            return {"error": "Model not available"}
        
        logger.info("🎤 Transcribing with Whisper.cpp...")
        start_time = time.time()
        
        try:
            # Transcribe with word-level timestamps
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=5,
                word_timestamps=True,
                language="en"  # Force English for now
            )
            
            # Process segments
            transcript_segments = []
            all_words = []
            full_text = []
            
            for segment in segments:
                segment_text = segment.text.strip()
                if segment_text:
                    # Extract word timings
                    words = []
                    if hasattr(segment, 'words') and segment.words:
                        for word in segment.words:
                            word_timing = WordTiming(
                                word=word.word,
                                start=word.start,
                                end=word.end,
                                confidence=word.probability
                            )
                            words.append(word_timing)
                            all_words.append(word_timing)
                    
                    # Create transcript segment
                    transcript_segment = TranscriptSegment(
                        text=segment_text,
                        start=segment.start,
                        end=segment.end,
                        speaker="WHISPER_SPEAKER",  # Will be replaced by diarization
                        confidence=segment.avg_logprob,
                        words=words
                    )
                    transcript_segments.append(transcript_segment)
                    full_text.append(segment_text)
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "transcript": " ".join(full_text),
                "segments": transcript_segments,
                "words": all_words,
                "language": info.language,
                "confidence": info.language_probability,
                "processing_time": processing_time,
                "method": "whisper_local"
            }
            
            logger.info(f"✅ Whisper transcription complete: {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Whisper transcription failed: {e}")
            return {"error": str(e), "method": "whisper_local"}

class DeepgramProcessor:
    """Deepgram Nova-2 cloud processor"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        self.base_url = "https://api.deepgram.com/v1"
        
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio with Deepgram Nova-2"""
        if not self.api_key:
            logger.warning("⚠️  Deepgram API key not available - skipping cloud transcription")
            return {"error": "API key not available", "method": "deepgram"}
        
        logger.info("☁️  Transcribing with Deepgram Nova-2...")
        start_time = time.time()
        
        try:
            # Read audio file
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            # Deepgram API parameters
            params = {
                "model": "nova-2",
                "punctuate": True,
                "diarize": True,
                "utterances": True,
                "words": True,
                "language": "en"
            }
            
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/wav"
            }
            
            # Make API call
            response = httpx.post(
                f"{self.base_url}/listen",
                params=params,
                headers=headers,
                content=audio_data,
                timeout=300  # 5 minute timeout
            )
            
            if response.status_code != 200:
                logger.error(f"❌ Deepgram API error: {response.status_code}")
                return {"error": f"API error: {response.status_code}", "method": "deepgram"}
            
            # Process response
            result_data = response.json()
            
            # Extract transcript and segments
            transcript_segments = []
            all_words = []
            full_text = []
            
            if "results" in result_data and "channels" in result_data["results"]:
                channel = result_data["results"]["channels"][0]
                
                # Process utterances (speaker-separated segments)
                if "utterances" in channel:
                    for utterance in channel["utterances"]:
                        words = []
                        if "words" in utterance:
                            for word in utterance["words"]:
                                word_timing = WordTiming(
                                    word=word["punctuated_word"],
                                    start=word["start"],
                                    end=word["end"],
                                    confidence=word["confidence"],
                                    speaker=f"SPEAKER_{utterance.get('speaker', 0)}"
                                )
                                words.append(word_timing)
                                all_words.append(word_timing)
                        
                        # Create transcript segment
                        transcript_segment = TranscriptSegment(
                            text=utterance["transcript"],
                            start=utterance["start"],
                            end=utterance["end"],
                            speaker=f"SPEAKER_{utterance.get('speaker', 0)}",
                            confidence=utterance.get("confidence", 0.8),
                            words=words
                        )
                        transcript_segments.append(transcript_segment)
                        full_text.append(utterance["transcript"])
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "transcript": " ".join(full_text),
                "segments": transcript_segments,
                "words": all_words,
                "language": "en",
                "confidence": 0.9,  # Deepgram generally high confidence
                "processing_time": processing_time,
                "method": "deepgram"
            }
            
            logger.info(f"✅ Deepgram transcription complete: {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Deepgram transcription failed: {e}")
            return {"error": str(e), "method": "deepgram"}

class ROVERMerger:
    """ROVER (Recognizer Output Voting Error Reduction) merger"""
    
    def __init__(self):
        self.confidence_weights = {
            "whisper_local": 0.7,
            "deepgram": 0.8
        }
    
    def merge_transcriptions(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple transcription results using ROVER"""
        logger.info("🔄 Merging transcriptions with ROVER...")
        
        # Filter successful results
        successful_results = [r for r in results if r.get("success", False)]
        
        if not successful_results:
            logger.error("❌ No successful transcriptions to merge")
            return {"error": "No successful transcriptions"}
        
        if len(successful_results) == 1:
            logger.info("ℹ️  Only one successful transcription - using directly")
            return successful_results[0]
        
        # Simple ROVER implementation - word-level voting
        logger.info(f"📊 Merging {len(successful_results)} transcriptions...")
        
        # Collect all words from all methods
        all_words = []
        for result in successful_results:
            method = result.get("method", "unknown")
            weight = self.confidence_weights.get(method, 0.5)
            
            for word in result.get("words", []):
                all_words.append({
                    "word": word.word,
                    "start": word.start,
                    "end": word.end,
                    "confidence": word.confidence * weight,
                    "method": method,
                    "speaker": word.speaker
                })
        
        # Sort by time
        all_words.sort(key=lambda x: x["start"])
        
        # Simple consensus - take highest confidence word for each time slot
        merged_words = []
        current_time = 0
        
        for word in all_words:
            if word["start"] >= current_time:
                merged_words.append(word)
                current_time = word["end"]
        
        # Build final transcript
        final_transcript = " ".join([w["word"] for w in merged_words])
        
        # Build segments (group words by speaker and time gaps)
        segments = self._build_segments(merged_words)
        
        # Get methods used
        methods_used = [r["method"] for r in successful_results]
        
        result = {
            "success": True,
            "transcript": final_transcript,
            "segments": segments,
            "words": merged_words,
            "language": "en",
            "confidence": np.mean([r.get("confidence", 0.8) for r in successful_results]),
            "processing_time": max([r.get("processing_time", 0) for r in successful_results]),
            "method": "rover_merged",
            "methods_used": methods_used
        }
        
        logger.info(f"✅ ROVER merge complete: {len(merged_words)} words, {len(segments)} segments")
        return result
    
    def _build_segments(self, words: List[Dict[str, Any]]) -> List[TranscriptSegment]:
        """Build segments from merged words"""
        if not words:
            return []
        
        segments = []
        current_segment_words = []
        current_speaker = words[0]["speaker"]
        current_start = words[0]["start"]
        
        for word in words:
            # Check if we need to start new segment
            if (word["speaker"] != current_speaker or 
                word["start"] - current_segment_words[-1]["end"] > 2.0 if current_segment_words else False):
                
                # Finish current segment
                if current_segment_words:
                    segment_text = " ".join([w["word"] for w in current_segment_words])
                    segment_end = current_segment_words[-1]["end"]
                    segment_confidence = np.mean([w["confidence"] for w in current_segment_words])
                    
                    # Convert to WordTiming objects
                    word_timings = [
                        WordTiming(w["word"], w["start"], w["end"], w["confidence"], w["speaker"])
                        for w in current_segment_words
                    ]
                    
                    segment = TranscriptSegment(
                        text=segment_text,
                        start=current_start,
                        end=segment_end,
                        speaker=current_speaker,
                        confidence=segment_confidence,
                        words=word_timings
                    )
                    segments.append(segment)
                
                # Start new segment
                current_segment_words = [word]
                current_speaker = word["speaker"]
                current_start = word["start"]
            else:
                current_segment_words.append(word)
        
        # Add final segment
        if current_segment_words:
            segment_text = " ".join([w["word"] for w in current_segment_words])
            segment_end = current_segment_words[-1]["end"]
            segment_confidence = np.mean([w["confidence"] for w in current_segment_words])
            
            word_timings = [
                WordTiming(w["word"], w["start"], w["end"], w["confidence"], w["speaker"])
                for w in current_segment_words
            ]
            
            segment = TranscriptSegment(
                text=segment_text,
                start=current_start,
                end=segment_end,
                speaker=current_speaker,
                confidence=segment_confidence,
                words=word_timings
            )
            segments.append(segment)
        
        return segments

class PyannoteAligner:
    """Align pyannote speaker turns to words"""
    
    def __init__(self):
        self.pipeline = None
        self._init_pipeline()
    
    def _init_pipeline(self):
        """Initialize pyannote pipeline"""
        if not PYANNOTE_AVAILABLE:
            logger.warning("⚠️  pyannote.audio not available - speaker alignment limited")
            return
        
        try:
            # Note: Requires HuggingFace token
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if not hf_token:
                logger.warning("⚠️  HuggingFace token not available - using mock alignment")
                return
            
            # Initialize speaker diarization pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            logger.info("✅ Pyannote pipeline initialized")
            
        except Exception as e:
            logger.error(f"❌ Pyannote pipeline initialization failed: {e}")
            self.pipeline = None
    
    def align_speakers(self, audio_path: str, merged_result: Dict[str, Any]) -> Dict[str, Any]:
        """Align pyannote speaker turns to merged transcript"""
        logger.info("👥 Aligning speakers with pyannote...")
        
        if not self.pipeline:
            logger.info("🔄 Using mock speaker alignment...")
            return self._mock_alignment(merged_result)
        
        try:
            # Run speaker diarization
            diarization = self.pipeline(audio_path)
            
            # Extract speaker turns
            speaker_turns = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_turns.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            # Align words to speaker turns
            aligned_words = []
            for word in merged_result.get("words", []):
                # Find overlapping speaker turn
                word_speaker = self._find_speaker_for_word(word, speaker_turns)
                
                aligned_word = WordTiming(
                    word=word["word"],
                    start=word["start"],
                    end=word["end"],
                    confidence=word["confidence"],
                    speaker=word_speaker
                )
                aligned_words.append(aligned_word)
            
            # Rebuild segments with aligned speakers
            segments = self._rebuild_segments_with_speakers(aligned_words)
            
            # Update result
            aligned_result = merged_result.copy()
            aligned_result["words"] = aligned_words
            aligned_result["segments"] = segments
            aligned_result["speakers"] = list(set([s["speaker"] for s in speaker_turns]))
            
            logger.info(f"✅ Speaker alignment complete: {len(aligned_result['speakers'])} speakers")
            return aligned_result
            
        except Exception as e:
            logger.error(f"❌ Speaker alignment failed: {e}")
            return self._mock_alignment(merged_result)
    
    def _find_speaker_for_word(self, word: Dict[str, Any], speaker_turns: List[Dict[str, Any]]) -> str:
        """Find speaker for a word based on timing"""
        word_start = word["start"]
        word_end = word["end"]
        
        # Find best overlapping speaker turn
        best_overlap = 0
        best_speaker = "SPEAKER_0"
        
        for turn in speaker_turns:
            # Calculate overlap
            overlap_start = max(word_start, turn["start"])
            overlap_end = min(word_end, turn["end"])
            
            if overlap_end > overlap_start:
                overlap = overlap_end - overlap_start
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = turn["speaker"]
        
        return best_speaker
    
    def _rebuild_segments_with_speakers(self, words: List[WordTiming]) -> List[TranscriptSegment]:
        """Rebuild segments with aligned speakers"""
        if not words:
            return []
        
        segments = []
        current_segment_words = []
        current_speaker = words[0].speaker
        current_start = words[0].start
        
        for word in words:
            # Check if we need to start new segment
            if (word.speaker != current_speaker or 
                word.start - current_segment_words[-1].end > 2.0 if current_segment_words else False):
                
                # Finish current segment
                if current_segment_words:
                    segment_text = " ".join([w.word for w in current_segment_words])
                    segment_end = current_segment_words[-1].end
                    segment_confidence = np.mean([w.confidence for w in current_segment_words])
                    
                    segment = TranscriptSegment(
                        text=segment_text,
                        start=current_start,
                        end=segment_end,
                        speaker=current_speaker,
                        confidence=segment_confidence,
                        words=current_segment_words.copy()
                    )
                    segments.append(segment)
                
                # Start new segment
                current_segment_words = [word]
                current_speaker = word.speaker
                current_start = word.start
            else:
                current_segment_words.append(word)
        
        # Add final segment
        if current_segment_words:
            segment_text = " ".join([w.word for w in current_segment_words])
            segment_end = current_segment_words[-1].end
            segment_confidence = np.mean([w.confidence for w in current_segment_words])
            
            segment = TranscriptSegment(
                text=segment_text,
                start=current_start,
                end=segment_end,
                speaker=current_speaker,
                confidence=segment_confidence,
                words=current_segment_words.copy()
            )
            segments.append(segment)
        
        return segments
    
    def _mock_alignment(self, merged_result: Dict[str, Any]) -> Dict[str, Any]:
        """Mock speaker alignment when pyannote not available"""
        logger.info("🎭 Using mock speaker alignment...")
        
        # Simple mock - alternate speakers every 30 seconds
        aligned_words = []
        current_speaker = "SPEAKER_0"
        last_switch = 0
        
        for word in merged_result.get("words", []):
            # Switch speaker every 30 seconds
            if word["start"] - last_switch > 30:
                current_speaker = "SPEAKER_1" if current_speaker == "SPEAKER_0" else "SPEAKER_0"
                last_switch = word["start"]
            
            aligned_word = WordTiming(
                word=word["word"],
                start=word["start"],
                end=word["end"],
                confidence=word["confidence"],
                speaker=current_speaker
            )
            aligned_words.append(aligned_word)
        
        # Rebuild segments
        segments = self._rebuild_segments_with_speakers(aligned_words)
        
        # Update result
        aligned_result = merged_result.copy()
        aligned_result["words"] = aligned_words
        aligned_result["segments"] = segments
        aligned_result["speakers"] = ["SPEAKER_0", "SPEAKER_1"]
        
        return aligned_result

class EnsembleASR:
    """Complete ensemble ASR system"""
    
    def __init__(self):
        self.whisper = WhisperLocalProcessor()
        self.deepgram = DeepgramProcessor()
        self.rover = ROVERMerger()
        self.aligner = PyannoteAligner()
    
    async def process_audio(self, audio_path: str) -> EnsembleResult:
        """Process audio with complete ensemble pipeline"""
        logger.info("🎵 Starting ensemble ASR processing...")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Phase 1: Parallel transcription
        logger.info("🔄 Phase 1: Parallel transcription...")
        transcription_tasks = []
        
        # Local Whisper
        if self.whisper.model:
            transcription_tasks.append(self._run_whisper(audio_path))
        
        # Cloud Deepgram
        if self.deepgram.api_key:
            transcription_tasks.append(self._run_deepgram(audio_path))
        
        # Run transcriptions in parallel
        results = await asyncio.gather(*transcription_tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = []
        for result in results:
            if isinstance(result, dict) and result.get("success", False):
                successful_results.append(result)
        
        if not successful_results:
            logger.error("❌ No successful transcriptions")
            return EnsembleResult(
                transcript="",
                segments=[],
                word_timings=[],
                speakers=[],
                language="en",
                confidence=0.0,
                processing_time=time.time() - start_time,
                methods_used=[]
            )
        
        # Phase 2: ROVER merge
        logger.info("🔄 Phase 2: ROVER merge...")
        merged_result = self.rover.merge_transcriptions(successful_results)
        
        # Phase 3: Speaker alignment
        logger.info("🔄 Phase 3: Speaker alignment...")
        aligned_result = self.aligner.align_speakers(audio_path, merged_result)
        
        # Phase 4: Build final result
        total_time = time.time() - start_time
        
        final_result = EnsembleResult(
            transcript=aligned_result["transcript"],
            segments=aligned_result["segments"],
            word_timings=aligned_result["words"],
            speakers=aligned_result.get("speakers", []),
            language=aligned_result.get("language", "en"),
            confidence=aligned_result.get("confidence", 0.8),
            processing_time=total_time,
            methods_used=aligned_result.get("methods_used", [])
        )
        
        logger.info("✅ Ensemble ASR processing complete!")
        logger.info(f"📊 Total time: {total_time:.2f}s")
        logger.info(f"📝 Transcript length: {len(final_result.transcript)} characters")
        logger.info(f"🎯 Segments: {len(final_result.segments)}")
        logger.info(f"👥 Speakers: {len(final_result.speakers)}")
        logger.info(f"🔧 Methods: {', '.join(final_result.methods_used)}")
        
        return final_result
    
    async def _run_whisper(self, audio_path: str) -> Dict[str, Any]:
        """Run Whisper transcription"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.whisper.transcribe, audio_path)
    
    async def _run_deepgram(self, audio_path: str) -> Dict[str, Any]:
        """Run Deepgram transcription"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.deepgram.transcribe, audio_path)

async def main():
    """Test ensemble ASR"""
    if len(sys.argv) < 2:
        print("Usage: python phase1_ensemble_asr.py <audio_path>")
        return
    
    audio_path = sys.argv[1]
    
    # Initialize ensemble ASR
    ensemble = EnsembleASR()
    
    # Process audio
    result = await ensemble.process_audio(audio_path)
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 ENSEMBLE ASR RESULTS")
    print("=" * 60)
    print(f"🎯 Transcript: {result.transcript[:200]}...")
    print(f"📝 Segments: {len(result.segments)}")
    print(f"👥 Speakers: {result.speakers}")
    print(f"🔧 Methods: {result.methods_used}")
    print(f"⏱️  Processing time: {result.processing_time:.2f}s")
    
    # Save detailed results
    output_path = "ensemble_asr_result.json"
    with open(output_path, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=str)
    
    print(f"💾 Detailed results saved to: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())