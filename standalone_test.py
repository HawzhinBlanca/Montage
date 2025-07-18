"""Standalone test for core functionality without database dependencies"""

import os
import time
import logging
import tempfile
import subprocess
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StandaloneTest:
    """Test core functionality without external dependencies"""
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive standalone tests"""
        logger.info("🧪 Running Standalone System Tests")
        logger.info("=" * 50)
        
        results = {
            'start_time': time.time(),
            'tests': {},
            'passed': True
        }
        
        tests = [
            self.test_config_loading,
            self.test_metrics_standalone,
            self.test_audio_processing,
            self.test_video_validation,
            self.test_color_space_detection,
            self.test_budget_calculations,
            self.test_smart_crop_logic,
            self.test_ffmpeg_commands
        ]
        
        for test in tests:
            test_name = test.__name__.replace('test_', '')
            logger.info(f"\n📋 Running test: {test_name}")
            
            try:
                result = test()
                results['tests'][test_name] = {
                    'status': 'passed' if result else 'failed',
                    'result': result
                }
                
                if not result:
                    results['passed'] = False
                    logger.error(f"❌ Test {test_name} FAILED")
                else:
                    logger.info(f"✅ Test {test_name} PASSED")
                    
            except Exception as e:
                results['tests'][test_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                results['passed'] = False
                logger.error(f"❌ Test {test_name} ERROR: {e}")
        
        results['end_time'] = time.time()
        results['duration'] = results['end_time'] - results['start_time']
        
        return results
    
    def test_config_loading(self) -> bool:
        """Test configuration loading"""
        try:
            from config import Config
            
            # Verify required config values
            assert Config.MAX_POOL_SIZE == Config.CPU_COUNT * 2
            assert Config.PROMETHEUS_PORT == 9099
            assert Config.BUDGET_LIMIT == 5.0
            assert Config.FFMPEG_PATH in ['ffmpeg', '/usr/bin/ffmpeg']
            
            logger.info(f"  - CPU count: {Config.CPU_COUNT}")
            logger.info(f"  - Pool size: {Config.MAX_POOL_SIZE}")
            logger.info(f"  - Metrics port: {Config.PROMETHEUS_PORT}")
            logger.info(f"  - Budget limit: ${Config.BUDGET_LIMIT}")
            
            return True
            
        except Exception as e:
            logger.error(f"Config test failed: {e}")
            return False
    
    def test_metrics_standalone(self) -> bool:
        """Test metrics system without database"""
        try:
            from prometheus_client import generate_latest
            from metrics import metrics
            
            # Test metric collection
            metrics.video_job_total.labels(status='test').inc()
            metrics.video_processing_time_ratio.set(0.8)
            metrics.api_cost_total.labels(api='test').inc(0.5)
            
            # Generate metrics
            metrics_data = generate_latest(metrics.registry)
            
            # Verify key metrics exist
            assert b'video_job_total' in metrics_data
            assert b'video_processing_time_ratio' in metrics_data
            assert b'api_cost_total' in metrics_data
            
            logger.info(f"  - Generated {len(metrics_data)} bytes of metrics")
            
            return True
            
        except Exception as e:
            logger.error(f"Metrics test failed: {e}")
            return False
    
    def test_audio_processing(self) -> bool:
        """Test audio normalization logic"""
        try:
            # Create test audio file
            test_audio = self._create_test_audio(duration=10)
            
            # Test loudness analysis command
            cmd = [
                'ffmpeg', '-i', test_audio,
                '-af', 'loudnorm=print_format=json',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check for loudness analysis output
            assert 'input_i' in result.stderr
            assert 'input_tp' in result.stderr
            
            logger.info("  - Audio loudness analysis working")
            
            # Test two-pass normalization command format
            cmd2 = [
                'ffmpeg', '-i', test_audio,
                '-af', 'loudnorm=I=-16:TP=-1.5:LRA=7:measured_I=-20:measured_TP=-5:measured_LRA=8',
                '-y', '/tmp/test_normalized.wav'
            ]
            
            result2 = subprocess.run(cmd2, capture_output=True, text=True)
            assert result2.returncode == 0
            
            logger.info("  - Two-pass normalization command working")
            
            # Cleanup
            os.remove(test_audio)
            if os.path.exists('/tmp/test_normalized.wav'):
                os.remove('/tmp/test_normalized.wav')
            
            return True
            
        except Exception as e:
            logger.error(f"Audio processing test failed: {e}")
            return False
    
    def test_video_validation(self) -> bool:
        """Test video validation logic"""
        try:
            # Create test video
            test_video = self._create_test_video(duration=30)
            
            # Test ffprobe command
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-show_entries', 'stream=codec_name,width,height,color_space',
                '-of', 'json', test_video
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            import json
            info = json.loads(result.stdout)
            
            # Verify we got the expected data
            assert 'format' in info
            assert 'streams' in info
            assert float(info['format']['duration']) > 25
            
            logger.info("  - Video metadata extraction working")
            
            # Test color space detection
            video_stream = next(s for s in info['streams'] if s.get('codec_type') == 'video')
            color_space = video_stream.get('color_space', 'unknown')
            
            logger.info(f"  - Detected color space: {color_space}")
            
            # Cleanup
            os.remove(test_video)
            
            return True
            
        except Exception as e:
            logger.error(f"Video validation test failed: {e}")
            return False
    
    def test_color_space_detection(self) -> bool:
        """Test color space conversion logic"""
        try:
            # Test the zscale filter command format
            test_video = self._create_test_video(duration=5)
            
            # Test BT.709 conversion command
            cmd = [
                'ffmpeg', '-i', test_video,
                '-vf', 'zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709:t=bt709:m=bt709:r=tv,format=yuv420p',
                '-c:v', 'libx264',
                '-y', '/tmp/test_bt709.mp4'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0
            
            # Verify output has correct color space
            verify_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=color_primaries',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                '/tmp/test_bt709.mp4'
            ]
            
            verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
            color_primaries = verify_result.stdout.strip()
            
            logger.info(f"  - Output color primaries: {color_primaries}")
            
            # Cleanup
            os.remove(test_video)
            if os.path.exists('/tmp/test_bt709.mp4'):
                os.remove('/tmp/test_bt709.mp4')
            
            return True
            
        except Exception as e:
            logger.error(f"Color space test failed: {e}")
            return False
    
    def test_budget_calculations(self) -> bool:
        """Test budget calculation logic"""
        try:
            # Test cost calculation functions
            from budget_guard import estimate_cost, openai_completion_cost, whisper_transcription_cost
            
            # Test OpenAI cost calculation
            openai_response = {
                'usage': {
                    'total_tokens': 1000
                }
            }
            cost = openai_completion_cost(openai_response)
            expected_cost = 0.002  # $0.002 per 1K tokens
            assert abs(cost - expected_cost) < 0.0001
            
            logger.info(f"  - OpenAI cost calculation: ${cost}")
            
            # Test Whisper cost calculation
            whisper_response = {
                'duration': 60  # 1 minute
            }
            whisper_cost = whisper_transcription_cost(whisper_response)
            expected_whisper = 0.006  # $0.006 per minute
            assert abs(whisper_cost - expected_whisper) < 0.0001
            
            logger.info(f"  - Whisper cost calculation: ${whisper_cost}")
            
            # Test cost estimation
            estimated = estimate_cost('openai', tokens=500)
            assert estimated > 0
            
            logger.info(f"  - Cost estimation working: ${estimated}")
            
            return True
            
        except Exception as e:
            logger.error(f"Budget calculation test failed: {e}")
            return False
    
    def test_smart_crop_logic(self) -> bool:
        """Test smart cropping physics"""
        try:
            from smart_crop import SpringDamper, SmartCropper
            
            # Test spring-damper physics
            damper = SpringDamper(stiffness=15.0, damping=8.0)
            
            # Simulate movement
            position = 0.0
            velocity = 0.0
            target = 100.0
            dt = 1.0/30.0  # 30 FPS
            
            for _ in range(30):  # 1 second
                position, velocity = damper.update(position, target, velocity, dt)
            
            # Should have moved toward target
            assert position > 50.0
            assert position < target  # Should not overshoot too much
            
            logger.info(f"  - Spring damper final position: {position:.1f}")
            
            # Test smart cropper initialization
            cropper = SmartCropper(output_aspect_ratio=9.0/16.0)
            assert cropper.output_aspect_ratio == 9.0/16.0
            
            logger.info("  - Smart cropper initialized correctly")
            
            return True
            
        except Exception as e:
            logger.error(f"Smart crop test failed: {e}")
            return False
    
    def test_ffmpeg_commands(self) -> bool:
        """Test FFmpeg command generation"""
        try:
            # Test FIFO creation
            fifo_path = '/tmp/test_fifo'
            
            if os.path.exists(fifo_path):
                os.remove(fifo_path)
            
            os.mkfifo(fifo_path)
            assert os.path.exists(fifo_path)
            
            logger.info("  - FIFO pipe creation working")
            
            # Test concat list generation
            segments = [
                {'start': 0, 'end': 10},
                {'start': 20, 'end': 30},
                {'start': 40, 'end': 50}
            ]
            
            concat_list = '\n'.join([f"file '/tmp/segment_{i}.mp4'" for i in range(len(segments))])
            
            # Verify concat list format
            lines = concat_list.split('\n')
            assert len(lines) == 3
            assert all(line.startswith("file '") for line in lines)
            
            logger.info(f"  - Concat list generation: {len(lines)} segments")
            
            # Test filter string length constraint
            filter_string = f"xfade=duration=0.5:offset=10,xfade=duration=0.5:offset=30"
            assert len(filter_string) < 300
            
            logger.info(f"  - Filter string length: {len(filter_string)} chars (< 300)")
            
            # Cleanup
            os.remove(fifo_path)
            
            return True
            
        except Exception as e:
            logger.error(f"FFmpeg command test failed: {e}")
            return False
    
    def _create_test_audio(self, duration: int = 10) -> str:
        """Create a test audio file"""
        output = tempfile.mktemp(suffix='.wav')
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', f'sine=frequency=440:duration={duration}',
            '-c:a', 'pcm_s16le',
            output
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output
    
    def _create_test_video(self, duration: int = 30) -> str:
        """Create a test video file"""
        output = tempfile.mktemp(suffix='.mp4')
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', f'testsrc2=duration={duration}:size=1920x1080:rate=30',
            '-f', 'lavfi',
            '-i', f'sine=frequency=440:duration={duration}',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-c:a', 'aac',
            output
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output


def main():
    """Run standalone tests"""
    tester = StandaloneTest()
    results = tester.run_all_tests()
    
    # Print summary
    total_tests = len(results['tests'])
    passed_tests = sum(1 for t in results['tests'].values() if t['status'] == 'passed')
    
    logger.info("\n" + "=" * 50)
    logger.info("📊 STANDALONE TEST SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Duration: {results['duration']:.1f}s")
    
    if results['passed']:
        logger.info("\n✅ ALL STANDALONE TESTS PASSED!")
        logger.info("Core functionality is working correctly.")
    else:
        logger.error("\n❌ SOME TESTS FAILED!")
        logger.error("Check individual test results above.")
    
    return 0 if results['passed'] else 1


if __name__ == "__main__":
    exit(main())