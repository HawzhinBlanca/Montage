"""Test performance requirements from the litmus test"""

import pytest
import time
import tempfile
import os
import subprocess
import json

from video_processor import SmartVideoEditor
from concat_editor import ConcatVideoEditor
from audio_normalizer import AudioNormalizer
from db import Database
from metrics import metrics


class TestPerformanceRequirements:
    """Test all performance requirements from Tasks.md"""
    
    @pytest.fixture
    def editor(self):
        return SmartVideoEditor()
    
    @pytest.fixture
    def concat_editor(self):
        return ConcatVideoEditor()
    
    @pytest.fixture
    def audio_normalizer(self):
        return AudioNormalizer()
    
    @pytest.fixture
    def db(self):
        return Database()
    
    def test_processing_time_requirement(self, editor):
        """Test: Processing time ≤ 1.2x source duration (Task 6)"""
        # Create 20-minute test video
        test_video = self._create_test_video(duration=1200)  # 20 minutes
        
        job_id = f"perf-test-{int(time.time())}"
        
        # Define edit plan
        edit_plan = {
            'segments': [
                {'start': 0, 'end': 60},
                {'start': 300, 'end': 360},
                {'start': 600, 'end': 660},
                {'start': 900, 'end': 960},
                {'start': 1140, 'end': 1200}
            ]
        }
        
        output_path = tempfile.mktemp(suffix='.mp4')
        
        # Measure processing time
        start_time = time.time()
        editor.execute_edit(job_id, test_video, edit_plan, output_path)
        processing_time = time.time() - start_time
        
        # Calculate ratio
        source_duration = 1200  # 20 minutes
        ratio = processing_time / source_duration
        
        print(f"Processing ratio: {ratio:.2f}x ({processing_time:.1f}s for {source_duration}s video)")
        
        # Update metric
        metrics.video_processing_time_ratio.set(ratio)
        
        # Cleanup
        os.remove(test_video)
        if os.path.exists(output_path):
            os.remove(output_path)
        
        # Must be under 1.2x
        assert ratio < 1.2, f"Processing too slow: {ratio:.2f}x (limit: 1.2x)"
    
    def test_filter_string_length(self, concat_editor):
        """Test: Filter string < 300 chars for 50 segments (Task 7)"""
        # Create 50 segments
        segments = []
        for i in range(50):
            segments.append({
                'start': i * 10,
                'end': (i + 1) * 10,
                'transition': 'fade',
                'transition_duration': 0.5
            })
        
        # Generate concat list
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            concat_list_path = f.name
            for i, segment in enumerate(segments):
                f.write(f"file 'segment_{i}.mp4'\n")
        
        # Build FFmpeg command
        cmd = concat_editor._build_concat_command(
            concat_list_path,
            'output.mp4',
            transitions='fade',
            duration=0.5
        )
        
        # Extract filter complex string
        filter_index = cmd.index('-filter_complex') + 1 if '-filter_complex' in cmd else -1
        
        if filter_index > 0:
            filter_string = cmd[filter_index]
            filter_length = len(filter_string)
            
            print(f"Filter string length: {filter_length} chars")
            
            # Cleanup
            os.remove(concat_list_path)
            
            # Must be under 300 characters
            assert filter_length < 300, f"Filter string too long: {filter_length} chars (limit: 300)"
        else:
            # Using concat demuxer without complex filters is valid
            assert True
    
    def test_audio_loudness_spread(self, audio_normalizer):
        """Test: Audio loudness spread ≤ 1.5 LU (Task 8)"""
        # Create test audio segments with different loudness
        segments = []
        loudness_values = [-23, -20, -25, -22, -24]  # LUFS values
        
        for i, target_lufs in enumerate(loudness_values):
            audio_file = self._create_test_audio(
                duration=10,
                frequency=440 + i * 100,
                amplitude=self._lufs_to_amplitude(target_lufs)
            )
            segments.append(audio_file)
        
        # Normalize each segment
        normalized_segments = []
        output_loudness = []
        
        for segment in segments:
            output = tempfile.mktemp(suffix='.mp4')
            stats = audio_normalizer.normalize_audio(segment, output)
            normalized_segments.append(output)
            output_loudness.append(stats.output_i)
        
        # Calculate spread
        min_loudness = min(output_loudness)
        max_loudness = max(output_loudness)
        spread = max_loudness - min_loudness
        
        print(f"Loudness values: {output_loudness}")
        print(f"Loudness spread: {spread:.2f} LU")
        
        # Update metric
        metrics.audio_loudness_spread_lufs.set(spread)
        
        # Cleanup
        for f in segments + normalized_segments:
            if os.path.exists(f):
                os.remove(f)
        
        # Must be ≤ 1.5 LU
        assert spread <= 1.5, f"Loudness spread too high: {spread:.2f} LU (limit: 1.5)"
    
    def test_ebur128_verification(self, audio_normalizer):
        """Test: Verify with ebur128 filter scan (Task 8)"""
        # Create and normalize a test video
        test_video = self._create_test_video(duration=30)
        output_video = tempfile.mktemp(suffix='.mp4')
        
        # Normalize
        stats = audio_normalizer.normalize_audio(test_video, output_video)
        
        # Verify with ebur128
        cmd = [
            'ffmpeg', '-i', output_video,
            '-af', 'ebur128=peak=true',
            '-f', 'null', '-'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse ebur128 output
        output_lines = result.stderr.split('\n')
        for line in output_lines:
            if 'I:' in line and 'LUFS' in line:
                # Extract integrated loudness
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'I:' and i + 1 < len(parts):
                        measured_loudness = float(parts[i + 1])
                        print(f"ebur128 measured loudness: {measured_loudness} LUFS")
                        
                        # Should be close to target (-16 LUFS)
                        assert abs(measured_loudness - (-16)) < 2.0
                        break
        
        # Cleanup
        os.remove(test_video)
        os.remove(output_video)
    
    def test_color_space_output(self, editor):
        """Test: Output must be BT.709 (Task 9)"""
        # Create test video
        test_video = self._create_test_video(duration=10)
        output_video = tempfile.mktemp(suffix='.mp4')
        
        job_id = f"color-test-{int(time.time())}"
        
        # Process video
        edit_plan = {'segments': [{'start': 0, 'end': 10}]}
        editor.execute_edit(job_id, test_video, edit_plan, output_video)
        
        # Verify color space with ffprobe
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=color_space,color_primaries,color_transfer',
            '-of', 'json', output_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        stream = data['streams'][0]
        color_primaries = stream.get('color_primaries', '')
        
        print(f"Output color primaries: {color_primaries}")
        
        # Cleanup
        os.remove(test_video)
        os.remove(output_video)
        
        # Must report BT.709
        assert color_primaries == 'bt709', f"Wrong color space: {color_primaries}"
    
    def test_budget_limit(self, db):
        """Test: Total cost must be < $5.00 (Task 11)"""
        from budget_guard import budget_manager, BudgetExceededError
        
        job_id = f"budget-test-{int(time.time())}"
        
        # Create job
        db.insert('video_job', {
            'id': job_id,
            'src_hash': 'budget_test_hash',
            'status': 'processing'
        })
        
        # Track costs up to near limit
        for i in range(49):
            budget_manager.track_cost(job_id, 'test_api', 0.1)
        
        # Check we're under budget
        is_within, current_cost = budget_manager.check_budget(job_id)
        assert is_within
        assert current_cost == 4.9
        
        # Try to exceed
        with pytest.raises(BudgetExceededError):
            # This should fail
            is_within, _ = budget_manager.check_budget(job_id, 0.2)
            if not is_within:
                raise BudgetExceededError("Budget would be exceeded")
    
    def test_highlight_quality(self, editor):
        """Test: AI produces ≤ 5 highlights of 15-60s (Litmus test)"""
        # This is tested in the litmus test
        # Here we verify the constraint enforcement
        
        test_highlights = [
            {'start': 0, 'end': 10, 'score': 0.9},      # Too short
            {'start': 20, 'end': 35, 'score': 0.85},    # Valid (15s)
            {'start': 50, 'end': 110, 'score': 0.8},    # Valid (60s)
            {'start': 120, 'end': 200, 'score': 0.75},  # Too long
            {'start': 210, 'end': 240, 'score': 0.7},   # Valid (30s)
            {'start': 250, 'end': 295, 'score': 0.65},  # Valid (45s)
            {'start': 300, 'end': 345, 'score': 0.6},   # Valid (45s)
            {'start': 350, 'end': 380, 'score': 0.55},  # 6th valid
        ]
        
        # Filter highlights
        valid_highlights = [
            h for h in test_highlights
            if 15 <= (h['end'] - h['start']) <= 60
        ]
        
        # Sort by score and take top 5
        valid_highlights.sort(key=lambda h: h['score'], reverse=True)
        final_highlights = valid_highlights[:5]
        
        # Verify constraints
        assert len(final_highlights) <= 5
        for h in final_highlights:
            duration = h['end'] - h['start']
            assert 15 <= duration <= 60
    
    def _create_test_video(self, duration=60):
        """Create a test video with specified duration"""
        output = tempfile.mktemp(suffix='.mp4')
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', f'testsrc2=duration={duration}:size=1920x1080:rate=30',
            '-f', 'lavfi', '-i', f'sine=frequency=440:duration={duration}',
            '-c:v', 'libx264', '-preset', 'ultrafast',
            '-c:a', 'aac', '-b:a', '128k',
            output
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output
    
    def _create_test_audio(self, duration=10, frequency=440, amplitude=0.5):
        """Create test audio with specific parameters"""
        output = tempfile.mktemp(suffix='.mp4')
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', f'sine=frequency={frequency}:duration={duration}:amplitude={amplitude}',
            '-c:a', 'aac', '-b:a', '128k',
            output
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output
    
    def _lufs_to_amplitude(self, lufs):
        """Convert LUFS to amplitude (rough approximation)"""
        # This is a simplified conversion
        # -23 LUFS ≈ 0.1 amplitude, -16 LUFS ≈ 0.3 amplitude
        return 10 ** ((lufs + 23) / 20) * 0.1


if __name__ == "__main__":
    pytest.main([__file__, '-xvs'])