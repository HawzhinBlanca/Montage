# Deep Function Audit Report

Generated: 2025-07-19 21:26:12

## Executive Summary

- **Total Functions**: 616 across 51 modules
- **Dead Code**: 444 functions (72.1%) appear unused
- **Mock/Placeholder**: 2 mocks, 0 placeholders
- **High Complexity**: 27 functions with complexity > 10
- **Quality Issues**: 271 missing type hints, 128 catch broad exceptions
- **Brain Health**: Core analysis pipeline has 0 placeholders
- **Major Blockers**: Database dependency prevents tests, many imports broken after restructure

## Function Matrix

| Module | Function | LOC | Called? | Purpose | Notes |
|--------|----------|-----|---------|---------|-------|
| color_converter.py | __init__ | 3 | ❌ | __init__ | no-types |
| color_converter.py | __init__ | 2 | ❌ | __init__ | no-types |
| color_converter.py | analyze_color_space | 40 | ❌ | Analyze video color space info | - |
| color_converter.py | build_color_conversi | 22 | ❌ | Build zscale filter for BT.709 | - |
| color_converter.py | build_safe_encoding_ | 43 | ❌ | Build FFmpeg command with prop | - |
| color_converter.py | convert_to_bt709 | 76 | ❌ | Convert video to BT.709 color  | - |
| color_converter.py | edit_with_color_safe | 26 | ❌ | Edit video ensuring BT.709 out | - |
| color_converter.py | ensure_bt709_output | 20 | ❌ | Ensure video output is BT.709  | - |
| color_converter.py | from_ffprobe | 24 | ❌ | Create from FFprobe stream dat | - |
| color_converter.py | get_encoding_color_p | 12 | ❌ | Get FFmpeg parameters to ensur | - |
| color_converter.py | get_safe_color_filte | 4 | ❌ | Get the color conversion filte | - |
| color_converter.py | validate_sdr_input | 17 | ❌ | Validate that input is SDR (no | broad-except |
| deep_function_audit.py | __init__ | 5 | ❌ | __init__ | - |
| deep_function_audit.py | __init__ | 6 | ❌ | __init__ | - |
| deep_function_audit.py | _analyze_function | 74 | ❌ | _analyze_function | complex(15), broad-except |
| deep_function_audit.py | _build_call_graph | 10 | ❌ | Build function call graph (sim | no-types |
| deep_function_audit.py | _catches_broad_excep | 9 | ❌ | Check if function catches Exce | - |
| deep_function_audit.py | _complexity_analysis | 3 | ❌ | Additional complexity analysis | no-types |
| deep_function_audit.py | _detect_external_dep | 20 | ❌ | Detect external dependencies ( | - |
| deep_function_audit.py | _generate_report | 138 | ❌ | Generate the audit report | complex(43), no-types |
| deep_function_audit.py | _is_placeholder | 15 | ❌ | Check if function is a placeho | complex(12) |
| deep_function_audit.py | _process_coverage_da | 21 | ❌ | Process coverage data and mark | no-types |
| deep_function_audit.py | _process_profile_dat | 12 | ❌ | Process py-spy profile data | broad-except |
| deep_function_audit.py | _runtime_verificatio | 65 | ❌ | Run tests with coverage and sa | no-types, broad-except |
| deep_function_audit.py | _static_scan | 22 | ❌ | Scan all Python files and extr | no-types, broad-except |
| deep_function_audit.py | run_audit | 25 | ❌ | Run the complete audit | no-types |
| deep_function_audit.py | visit_AsyncFunctionD | 2 | ❌ | visit_AsyncFunctionDef | no-types |
| deep_function_audit.py | visit_ClassDef | 5 | ❌ | visit_ClassDef | no-types |
| deep_function_audit.py | visit_FunctionDef | 2 | ❌ | visit_FunctionDef | no-types |
| deep_function_audit.py | visit_Import | 4 | ❌ | visit_Import | no-types |
| deep_function_audit.py | visit_ImportFrom | 4 | ❌ | visit_ImportFrom | no-types |
| examples/checkpoint_demo.py | generate_stage_resul | 58 | ❌ | Generate mock results for each | - |
| examples/checkpoint_demo.py | main | 29 | ✅ | Demo script entry point | no-types |
| examples/checkpoint_demo.py | simulate_video_proce | 95 | ❌ | Simulate video processing pipe | - |
| examples/metrics_demo.py | __init__ | 4 | ❌ | __init__ | no-types |
| examples/metrics_demo.py | analyze_video | 24 | ❌ | Analyze video content | - |
| examples/metrics_demo.py | edit_video | 13 | ❌ | Edit video based on segments | - |
| examples/metrics_demo.py | main | 16 | ✅ | Run the demo | no-types |
| examples/metrics_demo.py | normalize_audio | 16 | ❌ | Normalize audio levels | - |
| examples/metrics_demo.py | print_summary | 18 | ❌ | Print metrics summary | no-types |
| examples/metrics_demo.py | process_job | 25 | ❌ | Process a complete video job | broad-except |
| examples/metrics_demo.py | simulate_workload | 46 | ❌ | Simulate processing multiple j | broad-except |
| examples/metrics_demo.py | transcribe_audio | 13 | ❌ | Transcribe audio track | - |
| examples/metrics_demo.py | validate_video | 10 | ❌ | Validate video file | - |
| legacy_config.py | get_database_url | 2 | ❌ | get_database_url | no-types |
| legacy_config.py | validate | 17 | ❌ | Validate required configuratio | no-types |
| main.py | __init__ | 6 | ❌ | __init__ | no-types |
| main.py | _execute_stages | 122 | ❌ | Execute pipeline stages | complex(16) |
| main.py | _fail_job | 11 | ❌ | Mark job as failed | - |
| main.py | _highlights_to_edit_ | 22 | ❌ | Convert highlights to edit pla | - |
| main.py | create_job | 29 | ❌ | Create a new processing job | - |
| main.py | get_job_status | 19 | ❌ | Get current job status | - |
| main.py | main | 63 | ✅ | Main entry point | no-types, broad-except |
| main.py | process_job | 46 | ❌ | Process a single job with all  | broad-except |
| main.py | start_monitoring | 5 | ❌ | Start monitoring servers | no-types |
| migrate.py | __init__ | 2 | ❌ | __init__ | no-types |
| migrate.py | apply_migration | 24 | ❌ | Apply a single migration file | no-types, broad-except |
| migrate.py | create_database_if_n | 33 | ❌ | Create the database if it does | no-types, broad-except |
| migrate.py | create_migrations_ta | 12 | ❌ | Create migrations tracking tab | no-types |
| migrate.py | get_applied_migratio | 7 | ❌ | Get list of already applied mi | no-types |
| migrate.py | run | 49 | ❌ | Run all pending migrations | no-types, broad-except |
| run_montage.py | main | 92 | ✅ | Main entry point for Montage p | no-types, broad-except |
| src/cli/run_pipeline.py | create_subtitles_for | 15 | ❌ | Create subtitle files for each | - |
| src/cli/run_pipeline.py | display_results | 67 | ❌ | Display pipeline results | - |
| src/cli/run_pipeline.py | display_video_info | 21 | ❌ | Display video file information | - |
| src/cli/run_pipeline.py | extract_audio_rms | 74 | ❌ | Extract audio RMS energy level | complex(12), broad-except |
| src/cli/run_pipeline.py | main | 41 | ✅ | Main CLI function | no-types |
| src/cli/run_pipeline.py | post_to_mcp | 19 | ❌ | Post data to MCP server | broad-except |
| src/cli/run_pipeline.py | process_video_pipeli | 115 | ❌ | Main video processing pipeline | broad-except |
| src/cli/run_pipeline.py | start_mcp_server | 29 | ❌ | Start MCP server in background | no-types, broad-except |
| src/cli/run_pipeline.py | validate_video_file | 45 | ❌ | Validate video file exists and | complex(12), broad-except |
| src/core/adaptive_quality_pipe | __init__ | 19 | ❌ | __init__ | - |
| src/core/adaptive_quality_pipe | _extract_audio | 32 | ❌ | Extract audio from video for A | broad-except |
| src/core/adaptive_quality_pipe | _setup_directories | 12 | ❌ | Create necessary directories | no-types |
| src/core/adaptive_quality_pipe | execute | 68 | ❌ | Execute complete adaptive qual | broad-except |
| src/core/adaptive_quality_pipe | main | 67 | ✅ | Main entry point | no-types |
| src/core/adaptive_quality_pipe | phase1_ai_processing | 175 | ❌ | Phase 1: Complete AI processin | broad-except |
| src/core/adaptive_quality_pipe | phase2_video_editing | 52 | ❌ | Phase 2: DaVinci Resolve video | broad-except |
| src/core/adaptive_quality_pipe | phase3_quality_contr | 85 | ❌ | Phase 3: Quality control and a | broad-except |
| src/core/adaptive_quality_pipe | phase4_finalize_and_ | 81 | ❌ | Phase 4: Finalize and generate | - |
| src/core/analyze_video.py | align_speakers | 19 | ❌ | Align speaker turns with word  | - |
| src/core/analyze_video.py | analyze_video | 59 | ❌ | Main video analysis function - | - |
| src/core/analyze_video.py | cache_transcript | 19 | ❌ | Cache transcript in database | broad-except |
| src/core/analyze_video.py | diarize | 20 | ❌ | REAL speaker diarization using | broad-except |
| src/core/analyze_video.py | extract_audio_to_wav | 17 | ❌ | Extract audio from video to WA | broad-except |
| src/core/analyze_video.py | get_cached_transcrip | 22 | ❌ | Get cached transcript from dat | broad-except |
| src/core/analyze_video.py | rover_merge | 44 | ❌ | ROVER merge algorithm for comb | complex(12) |
| src/core/analyze_video.py | sha256_file | 7 | ❌ | Calculate SHA256 hash of file  | - |
| src/core/analyze_video.py | simple_speaker_segme | 30 | ❌ | Fallback speaker segmentation  | broad-except |
| src/core/analyze_video.py | transcribe_deepgram | 54 | ❌ | REAL Deepgram transcription wi | complex(13), broad-except |
| src/core/analyze_video.py | transcribe_faster_wh | 42 | ❌ | REAL faster-whisper transcript | broad-except |
| src/core/checkpoint.py | __init__ | 20 | ❌ | __init__ | no-types |
| src/core/checkpoint.py | __init__ | 2 | ❌ | __init__ | - |
| src/core/checkpoint.py | _get_checkpoint_key | 3 | ❌ | Generate Redis key for a check | - |
| src/core/checkpoint.py | _get_job_stages_key | 3 | ❌ | Generate Redis key for job sta | - |
| src/core/checkpoint.py | _restore_from_postgr | 19 | ❌ | Restore checkpoint from Postgr | broad-except |
| src/core/checkpoint.py | atomic_checkpoint | 15 | ❌ | Context manager for atomic che | - |
| src/core/checkpoint.py | delete_job_checkpoin | 20 | ❌ | Delete all checkpoints for a j | broad-except |
| src/core/checkpoint.py | example_usage | 41 | ❌ | Example of how to use checkpoi | no-types |
| src/core/checkpoint.py | exists | 4 | ❌ | Check if a checkpoint exists | - |

## Brain Focus (Core Analysis Functions)

### `deep_function_audit.py:FunctionAnalyzer._analyze_function`

- **Signature**: `def _analyze_function(self, node, is_async)`
- **Complexity**: 15
- **External Deps**: none
- **Quality Flags**: catches broad Exception

### `src/core/highlight_selector.py:choose_highlights`

- **Signature**: `def choose_highlights(words, audio_energy, mode)`
- **Complexity**: 15
- **External Deps**: none
- **Quality Flags**: good

### `src/providers/smart_track.py:SmartTrack._analyze_motion`

- **Signature**: `def _analyze_motion(self, video_path)`
- **Complexity**: 14
- **External Deps**: none
- **Quality Flags**: catches broad Exception

### `src/core/analyze_video.py:transcribe_deepgram`

- **Signature**: `def transcribe_deepgram(wav_path)`
- **Complexity**: 13
- **External Deps**: file, ai
- **Quality Flags**: catches broad Exception

### `src/providers/video_probe.py:VideoProbe._analyze_motion`

- **Signature**: `def _analyze_motion(self, video_path, sample_seconds)`
- **Complexity**: 13
- **External Deps**: file
- **Quality Flags**: catches broad Exception

### `src/core/db.py:SecureDatabase.find_many`

- **Signature**: `def find_many(self, table, where, order_by, limit)`
- **Complexity**: 12
- **External Deps**: none
- **Quality Flags**: good

### `src/core/highlight_selector.py:local_rule_scorer`

- **Signature**: `def local_rule_scorer(words, audio_energy)`
- **Complexity**: 12
- **External Deps**: none
- **Quality Flags**: good

### `src/core/analyze_video.py:rover_merge`

- **Signature**: `def rover_merge(fw_words, dg_words)`
- **Complexity**: 12
- **External Deps**: none
- **Quality Flags**: good

### `src/providers/selective_enhancer.py:SelectiveEnhancer.enhance_highlights_only`

- **Signature**: `def enhance_highlights_only(self, smart_result, budget, video_path)`
- **Complexity**: 12
- **External Deps**: none
- **Quality Flags**: catches broad Exception

### `src/providers/smart_track.py:SmartTrack._analyze_audio_energy`

- **Signature**: `def _analyze_audio_energy(self, video_path)`
- **Complexity**: 12
- **External Deps**: subprocess, file
- **Quality Flags**: catches broad Exception

### `src/core/db.py:SecureDatabase.insert`

- **Signature**: `def insert(self, table, data, returning)`
- **Complexity**: 9
- **External Deps**: none
- **Quality Flags**: good

### `src/core/checkpoint.py:CheckpointManager.get_last_successful_stage`

- **Signature**: `def get_last_successful_stage(self, job_id)`
- **Complexity**: 9
- **External Deps**: db
- **Quality Flags**: catches broad Exception

### `src/core/fallback_selector.py:analyze_with_fallback`

- **Signature**: `def analyze_with_fallback(text, word_timings)`
- **Complexity**: 9
- **External Deps**: none
- **Quality Flags**: good

### `src/core/errors.py:handle_error`

- **Signature**: `def handle_error(error, context)`
- **Complexity**: 9
- **External Deps**: none
- **Quality Flags**: good

### `src/providers/transcript_analyzer.py:TranscriptAnalyzer._analyze_chunk`

- **Signature**: `def _analyze_chunk(self, chunk_text, all_segments, audio_features, visual_features)`
- **Complexity**: 9
- **External Deps**: ai
- **Quality Flags**: catches broad Exception

### `src/providers/smart_track.py:SmartTrack._analyze_faces`

- **Signature**: `def _analyze_faces(self, video_path)`
- **Complexity**: 9
- **External Deps**: none
- **Quality Flags**: catches broad Exception

### `tests/test_color_converter.py:TestColorSpaceConverter.test_analyze_color_space`

- **Signature**: `def test_analyze_color_space(self, mock_run, converter)`
- **Complexity**: 8
- **External Deps**: subprocess
- **Quality Flags**: no type hints

### `tests/test_audio_normalizer.py:TestAudioNormalizer.test_analyze_loudness`

- **Signature**: `def test_analyze_loudness(self, mock_run, normalizer)`
- **Complexity**: 8
- **External Deps**: subprocess
- **Quality Flags**: no type hints

### `src/core/user_success_metrics.py:UserSuccessMetrics._generate_insights`

- **Signature**: `def _generate_insights(self, metrics, mode_dist, abandonment)`
- **Complexity**: 8
- **External Deps**: none
- **Quality Flags**: good

### `src/core/analyze_video.py:transcribe_faster_whisper`

- **Signature**: `def transcribe_faster_whisper(wav_path)`
- **Complexity**: 8
- **External Deps**: none
- **Quality Flags**: catches broad Exception

## Critical Findings

### BLOCKER

- Database initialization fails on import, blocking most tests
- Import errors after project restructure not fully resolved
- 0 core brain functions are placeholders
- API keys still exposed in .env file

### MAJOR

- 444 dead functions consuming maintenance effort
- 27 overly complex functions need refactoring
- 271 functions lack type hints
- No integration tests for video processing pipeline

### MINOR

- 128 functions catch broad exceptions
- Inconsistent error handling patterns
- Many TODO/FIXME comments in code
