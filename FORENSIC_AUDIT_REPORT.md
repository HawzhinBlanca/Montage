# Forensic Code Audit Report: Montage Video Processing System

**Audit Date:** 2025-07-23  
**Auditor:** Forensic Code Analysis  
**Objective:** Determine which features actually work vs fake/placeholders

---

## Executive Summary

The Montage codebase shows a pattern of **mixed implementation quality**: some features are fully functional with sophisticated implementations, while others are fake or have been removed. Most notably, several "AI" features are either placeholders or not integrated into the actual pipeline.

---

## 1. SMART CROP SYSTEM

**VERDICT: REAL** ✅

### Evidence
- **File:** `/src/utils/intelligent_crop.py`
- **Implementation:** Uses OpenCV Haar Cascades for face detection
- **Execution Flow:**
  1. Loads OpenCV cascade classifiers (frontal + profile)
  2. Detects faces with multiple scale factors (1.02, 1.05, 1.1)
  3. Falls back to motion detection using optical flow
  4. Returns weighted face positions or center crop (0.5, 0.4)

### Real Implementation Code
```python
faces = self.face_cascade.detectMultiScale(
    gray, scaleFactor=scale_factor, minNeighbors=3, 
    minSize=(20, 20)
)
# ... weights faces by size, removes duplicates
```

### Limitations
- Missing DNN models (TensorFlow face detector files not present)
- MediaPipe not integrated (only in debug scripts)
- Tests don't verify actual face coordinates
- **But it DOES return different crop coordinates for different face positions**

---

## 2. AUDIO NORMALIZATION

**VERDICT: REAL CODE, NOT USED** ⚠️

### Evidence
- **File:** `/src/providers/audio_normalizer.py`
- **Implementation:** Full EBU R128 loudnorm with FFmpeg
- **Problem:** NOT integrated into main pipeline!

### Real Implementation Code
```python
# Two-pass loudnorm process
def _apply_normalization(self, input_file: str, output_file: str, 
                        measured_i: float, measured_tp: float, measured_lra: float):
    audio_filter = (
        f'loudnorm=I={self.target_i}:TP={self.target_tp}:LRA={self.target_lra}:'
        f'measured_I={measured_i}:measured_TP={measured_tp}:'
        f'measured_LRA={measured_lra}:measured_thresh={measured_thresh}:'
        f'offset={offset}:linear=true:print_format=json'
    )
```

### Critical Finding
- ✅ Algorithm is real (proper LUFS measurement)
- ✅ Test coverage shows different inputs → different normalization
- ❌ **Never called in main video pipeline**
- Videos processed with basic `-c:a aac` without normalization

---

## 3. SPEAKER DIARIZATION

**VERDICT: FAKE** ❌

### Evidence
- **File:** `/src/core/analyze_video.py`
- **Implementation:** Alternating speaker IDs on timer

### Fake Implementation Code
```python
# Line 562: Simply alternates speakers
speaker_id = (speaker_id + 1) % 2

# Line 606: Fixed segments with rotating IDs
speaker_id = (speaker_id + 1) % 3  # Simulate 3 speakers
```

### Real vs Fake
- ✅ Deepgram API called with `diarize=True` (could be real if API returns speakers)
- ❌ Fallback is pure fake: alternates SPEAKER_0/SPEAKER_1 every segment
- ❌ pyannote.audio in requirements but NEVER imported or used
- ❌ No voice embeddings, no clustering, no actual voice analysis
- **Same voice does NOT get same speaker ID across segments**

---

## 4. MEMORY MANAGEMENT

**VERDICT: REAL** ✅

### Evidence
- **Files:** `/src/utils/memory_manager.py`, `/src/utils/resource_manager.py`
- **Implementation:** Full psutil-based monitoring with enforcement

### Real Implementation Code
```python
# Actual system memory reading
memory = psutil.virtual_memory()
available_mb = memory.available / (1024 * 1024)

# Real process termination
if mem_usage > soft_limit_mb:
    process.terminate()
    process.wait(timeout=5)
    if process.poll() is None:
        process.kill()
```

### Behavior Changes Based on Memory
- **70-80% usage:** Triggers garbage collection
- **80-90% usage:** Kills oldest FFmpeg process
- **>90% usage:** Emergency mode, kills all FFmpeg
- Processing modes adapt: FULL_QUALITY → BALANCED → SURVIVAL
- **Memory usage DOES affect behavior**

---

## 5. EMOTION ANALYSIS

**VERDICT: REMOVED (was FAKE)** ❌

### Evidence
- **File:** `emotion_analyzer.py` - DELETED
- **Comment:** "Removed all fake AI components: emotion_analyzer"
- **Audit:** "❌ Emotion Analyzer (returns random scores)"

### Current State
- No emotion analysis in codebase
- `emotion_energy_proxy` in outputs is just `score * 100`
- No emotion models, no sentiment analysis, no facial emotion
- Gemini prompt mentions emotions but doesn't analyze them
- **Identical inputs would NOT produce identical emotion scores (no emotion scoring exists)**

---

## Summary Matrix

| Feature | Status | Evidence | Behavior |
|---------|--------|----------|----------|
| **Smart Crop** | ✅ REAL | OpenCV face detection | Different faces → different crops |
| **Audio Norm** | ⚠️ REAL, UNUSED | EBU R128 implementation | Not integrated in pipeline |
| **Diarization** | ❌ FAKE | Alternating IDs | Same voice ≠ same speaker ID |
| **Memory Mgmt** | ✅ REAL | psutil monitoring | Memory limits enforced |
| **Emotion** | ❌ REMOVED | Was fake, now deleted | No emotion analysis exists |

---

## Patterns Observed

1. **"Build it but don't wire it"**: Audio normalization is fully implemented but not connected
2. **"Fake it till you make it"**: Speaker diarization has elaborate fallbacks that just alternate IDs
3. **"Delete the evidence"**: Fake emotion analyzer was removed rather than fixed
4. **"Real infrastructure"**: Memory management and resource handling are genuinely sophisticated
5. **"Mixed quality"**: Some features (face detection, memory) are real; others (diarization, emotion) are fake

The codebase appears to be in transition from a demo/prototype (with fake features) to a real system (with working implementations), but the integration is incomplete.