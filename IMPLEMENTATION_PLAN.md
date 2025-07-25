# Montage Pipeline - Professional Implementation Plan
## Total Timeline: 6 Weeks | Start: 2024-01-24

---

## PHASE 0: UNBLOCK CORE (1-2 Days)
**Objective**: Make pipeline functional with zero TODOs and proper CI

### Task 0.1: Implement CLI execute_plan() [2 hours]
**File**: `montage/cli/run_pipeline.py:866`
```python
def execute_plan_from_cli(plan_path: str, output_path: str):
    try:
        with open(plan_path, 'r') as f:
            plan = json.load(f)
        source = plan["source_video_path"]
        clips  = plan["clips"]
        editor = VideoEditor(source_path=source)
        editor.process_clips(clips, output_path)
        logger.info(f"Executed plan: {plan_path} → {output_path}")
    except Exception as e:
        logger.critical(f"execute_plan failed: {e}")
        sys.exit(1)
```

**Verification**:
1. Run: `python -m montage.cli.run_pipeline tests/assets/minimal.json -o out.mp4`
2. Check: Exit code must be 0
3. Check: Log must show "Executed plan: tests/assets/minimal.json → out.mp4"
4. Check: `ffprobe -v error -show_entries format=duration out.mp4` shows valid duration

### Task 0.2: Fix CI Environment [1 hour]
**File**: `.github/workflows/ci.yml`
```yaml
env:
  JWT_SECRET_KEY: "test-key-for-ci"
```

**Verification**:
1. Check: First line of pytest shows "Using JWT_SECRET_KEY=test-key-for-ci"
2. Run: `pytest --version` in CI must succeed
3. Check: No JWT-related errors in CI logs

### Task 0.3: Pin NumPy Version [30 min]
**File**: `requirements.txt`
```
numpy<2.0
```

**Verification**:
1. Run: `pip install -r requirements.txt`
2. Check: `pip show numpy | grep Version` shows < 2.0
3. Check: No NumPy compatibility warnings
4. Run: `python -c "import numpy; print(numpy.__version__)"` < 2.0

### Task 0.4: End-to-End CLI Validation [1 hour]
**Commands**:
```bash
python -m montage.cli.run_pipeline tests/assets/minimal.json -o out.mp4
echo "Exit code: $?"
ffprobe -v error -show_entries format=duration out.mp4
```

**Verification**:
1. Exit code: 0
2. FFprobe duration matches fixture duration ±0.1s
3. Output file size > 0
4. No ERROR logs during execution

### Task 0.5: Remove All TODOs [2 hours]
**Commands**:
```bash
grep -R "TODO" montage/ | grep -v "__pycache__" > todos.txt
# Fix each TODO
grep -R "TODO" montage/ | wc -l  # Must be 0
```

**Verification**:
1. `grep -R "TODO" montage/ | wc -l` returns 0
2. All removed TODOs have proper implementations
3. All tests still pass after TODO removal

---

## PHASE 1: VALIDATE & HARDEN CORE (1 Week)

### Task 1.1: Audio Normalization Tests [4 hours]
**File**: `tests/test_audio_norm.py`
```python
def test_audio_normalization_applied():
    normalizer = AudioNormalizer()
    result = normalizer.normalize_audio("input.mp4", "output.mp4")
    assert result.last_analysis.input_i != result.last_analysis.output_i
    assert abs(result.last_analysis.output_i - (-16.0)) < 1.0  # Target LUFS
```

**Verification**:
1. Test passes in CI
2. Manual test: Input audio -23 LUFS → Output -16±1 LUFS
3. `ffmpeg -i output.mp4 -af ebur128 -f null -` confirms normalization

### Task 1.2: Smart Crop Verification [4 hours]
**File**: `tests/test_smart_crop.py`
```python
def test_face_detection_crop():
    result = create_intelligent_vertical_video("face_video.mp4", "9:16")
    assert result.faces_detected > 0
    assert result.crop_params.x > 0 or result.crop_params.y > 0
    assert result.output_aspect_ratio == "9:16"
```

**Verification**:
1. Test with video containing faces
2. Output video shows face-centered crop
3. Aspect ratio exactly 9:16 (1080x1920)

### Task 1.3: Memory Management Tests [6 hours]
**File**: `tests/test_memory_limits.py`
```python
def test_memory_pressure_response():
    manager = MemoryManager()
    # Simulate high memory
    manager._simulate_pressure(MemoryPressureLevel.HIGH)
    config = manager.get_adaptive_config()
    assert config["max_workers"] < 4  # Reduced from default
    assert config["chunk_size_mb"] < 1024  # Reduced chunk size
```

**Verification**:
1. Run with 2GB video file
2. Monitor: `ps aux | grep python` shows < 8GB RAM usage
3. No OOM errors during processing
4. Graceful degradation under pressure

### Task 1.4: Database Schema Migration [4 hours]
**Actions**:
1. Add missing columns to allowed_columns in db.py
2. Create migration script for src_hash, codec, stage, error_message
3. Test with both PostgreSQL and in-memory fallback

**Verification**:
1. No "Invalid column name" warnings
2. Job persistence works across restarts
3. Checkpoint save/load succeeds

### Task 1.5: Error Propagation Tests [4 hours]
**File**: `tests/test_error_handling.py`
```python
def test_pipeline_error_propagation():
    with pytest.raises(PipelineError) as exc:
        pipeline.process_job("invalid_video.mp4")
    assert exc.value.stage == "validation"
    assert "does not exist" in str(exc.value)
```

**Verification**:
1. Invalid input → Clear error message
2. Partial failure → Checkpoint recovery works
3. All errors logged with context

---

## PHASE 2: COMPLETE VISUAL INTELLIGENCE (1 Week)

### Task 2.1: Face Detection Integration [8 hours]
**Verification**:
1. Test with 10 videos, detect faces in 90%+
2. Bounding boxes accurate within 5% IoU
3. Performance: < 1s per frame on M4 Max

### Task 2.2: ROVER Implementation [8 hours]
**Verification**:
1. Merge 3 transcripts → Higher confidence than any single
2. O(n log n) performance verified
3. Word error rate reduced by 15%+

### Task 2.3: Speaker Diarization [8 hours]
**Verification**:
1. 2-speaker video: 85%+ accuracy
2. Speaker changes detected within 1s
3. Consistent speaker IDs across segments

### Task 2.4: Scene Ranking with Gemma [6 hours]
**Verification**:
1. Ollama gemma3:latest responds in < 2s
2. Rankings correlate with manual ratings (0.7+ correlation)
3. Fallback to local scoring works

### Task 2.5: Story Beat Detection [6 hours]
**Verification**:
1. Detect hook/climax/resolution in 80%+ videos
2. Beat timestamps align with content changes
3. API costs < $0.01 per video

---

## PHASE 3: FULL PIPELINE INTEGRATION (3-4 Days)

### Task 3.1: Checkpoint Recovery [6 hours]
**Verification**:
1. Kill at any stage → Resumes correctly
2. No duplicate processing
3. State consistency maintained

### Task 3.2: Cost Tracking [4 hours]
**Verification**:
1. API costs logged per job
2. Budget limits enforced
3. Cost report generation works

### Task 3.3: Progress Reporting [4 hours]
**Verification**:
1. Real-time progress via WebSocket
2. ETA calculations within 20% accuracy
3. Stage transitions logged

### Task 3.4: Multi-format Support [6 hours]
**Verification**:
1. Process: MP4, MOV, AVI, MKV, WebM
2. Output: H.264, H.265, VP9
3. Codec errors handled gracefully

---

## PHASE 4: ADVANCED FEATURES (1 Week)

### Task 4.1: Scene Detection (PySceneDetect) [8 hours]
**Verification**:
1. Detect cuts with 95%+ accuracy
2. Adaptive threshold based on content
3. Scene list exportable as JSON

### Task 4.2: MMTracking Integration [12 hours]
**Verification**:
1. Track subjects across scenes
2. Re-identification accuracy > 80%
3. Real-time performance (25+ FPS)

### Task 4.3: CLIP Visual Search [8 hours]
**Verification**:
1. Find similar scenes in < 1s
2. Text-to-scene search works
3. Embedding cache reduces compute

### Task 4.4: Motion Stabilization [8 hours]
**Verification**:
1. Shaky footage → Smooth output
2. No warping artifacts
3. Processing < 2x realtime

---

## PHASE 5: AI ORCHESTRATION (1 Week)

### Task 5.1: GPT-4V Integration [12 hours]
**Verification**:
1. Describe scenes accurately
2. Generate editing suggestions
3. Cost < $0.10 per video

### Task 5.2: Narrative Arc Detection [8 hours]
**Verification**:
1. Identify story structure in 70%+ videos
2. Suggest edit points based on narrative
3. Maintain story coherence

### Task 5.3: Multi-modal Scoring [8 hours]
**Verification**:
1. Combine audio/video/text signals
2. Scores correlate with human preference
3. Explainable ranking reasons

### Task 5.4: Auto-editing Rules [8 hours]
**Verification**:
1. Apply style templates correctly
2. Respect pacing preferences
3. Output matches brief requirements

---

## PHASE 6: PRODUCTION READINESS (3-4 Days)

### Task 6.1: Load Testing [8 hours]
**Verification**:
1. Handle 10 concurrent jobs
2. No memory leaks over 24 hours
3. Graceful degradation at limit

### Task 6.2: Monitoring & Alerts [6 hours]
**Verification**:
1. Prometheus metrics exported
2. Alert on failures/high latency
3. Grafana dashboard functional

### Task 6.3: Documentation [6 hours]
**Verification**:
1. API docs auto-generated
2. Setup guide < 10 min to working
3. Architecture diagrams current

### Task 6.4: Docker & K8s [8 hours]
**Verification**:
1. Docker build < 5 minutes
2. K8s deployment scales 1-10 pods
3. Health checks pass consistently

---

## DAILY VERIFICATION PROTOCOL

1. **Morning Standup Check**:
   - Review yesterday's task completion
   - Run full test suite
   - Check for regressions

2. **Task Completion Criteria**:
   - Code implemented exactly as specified
   - All verification steps pass
   - No new TODOs introduced
   - Test coverage maintained/increased

3. **Evening Validation**:
   - End-to-end pipeline test
   - Memory/CPU usage within limits
   - All logs clean (no unexpected warnings/errors)

4. **Sign-off Requirements**:
   - Peer review via PR
   - CI/CD pipeline green
   - Documentation updated
   - Verification screenshots/logs archived

---

## SUCCESS METRICS

- **Phase 0**: Pipeline runs without errors, CI green
- **Phase 1**: All core features tested and hardened
- **Phase 2**: Visual intelligence accuracy > 85%
- **Phase 3**: Full pipeline processes 10min video in < 5min
- **Phase 4**: Advanced features add < 20% processing time
- **Phase 5**: AI suggestions improve output quality by 30%
- **Phase 6**: Production deployment handles 100 videos/day

---

## RISK MITIGATION

1. **If behind schedule**: Prioritize core features, defer advanced
2. **If tests fail**: Stop new development until fixed
3. **If performance degrades**: Profile and optimize before continuing
4. **If costs exceed budget**: Implement stricter limits, use more local models

---

This plan ensures 100% implementation accuracy with rigorous verification at each step.