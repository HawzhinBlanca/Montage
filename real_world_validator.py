"""Real-world video validation for Phase 3 Task 16"""

import os
import time
import logging
import json
import subprocess
import tempfile
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import hashlib
import cv2
import numpy as np

from main import VideoProcessingPipeline
from litmus_test import LitmusTest
from db import Database
from metrics import metrics
from monitoring_integration import MonitoringServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealWorldValidator:
    """Validates the system with real 45-minute videos"""
    
    def __init__(self):
        self.pipeline = VideoProcessingPipeline()
        self.litmus = LitmusTest()
        self.db = Database()
        self.monitoring = MonitoringServer()
        self.test_results = {
            'start_time': datetime.utcnow().isoformat(),
            'videos_tested': [],
            'summary': {},
            'passed': True
        }
        
    def run_real_world_validation(self, video_sources: List[str]) -> Dict[str, Any]:
        """Run comprehensive real-world validation"""
        logger.info("🎬 Starting Real-World Video Validation")
        logger.info("=" * 60)
        
        # Start monitoring
        self.monitoring.start()
        time.sleep(2)
        
        # Process each test video
        for i, video_source in enumerate(video_sources, 1):
            logger.info(f"\n📹 Processing Video {i}/{len(video_sources)}: {video_source}")
            logger.info("-" * 50)
            
            try:
                video_result = self._test_single_video(video_source, f"real-world-{i}")
                self.test_results['videos_tested'].append(video_result)
                
                if not video_result['passed']:
                    self.test_results['passed'] = False
                    logger.error(f"❌ Video {i} failed validation")
                else:
                    logger.info(f"✅ Video {i} passed validation")
                    
            except Exception as e:
                logger.error(f"❌ Video {i} processing error: {e}")
                self.test_results['videos_tested'].append({
                    'video_source': video_source,
                    'status': 'error',
                    'error': str(e),
                    'passed': False
                })
                self.test_results['passed'] = False
        
        # Generate final summary
        self._generate_summary()
        
        return self.test_results
    
    def _test_single_video(self, video_source: str, test_id: str) -> Dict[str, Any]:
        """Test a single video through the complete pipeline"""
        start_time = time.time()
        
        # Download or prepare video
        video_path = self._prepare_video(video_source)
        
        # Validate input meets 45-minute criteria
        video_info = self._validate_input_video(video_path)
        if not video_info['valid']:
            return {
                'video_source': video_source,
                'status': 'invalid_input',
                'reason': video_info['reason'],
                'passed': False
            }
        
        # Create job and process
        output_path = f"/tmp/real_world_output_{test_id}.mp4"
        
        try:
            job_id = self.pipeline.create_job(
                input_path=video_path,
                output_path=output_path,
                options={
                    'smart_crop': True,
                    'aspect_ratio': '9:16'
                }
            )
            
            # Process with full pipeline
            processing_start = time.time()
            result = self.pipeline.process_job(job_id)
            processing_time = time.time() - processing_start
            
            # Validate all litmus test criteria
            validation_results = self._validate_litmus_criteria(
                video_path, output_path, job_id, processing_time
            )
            
            # Calculate final score
            total_time = time.time() - start_time
            
            return {
                'video_source': video_source,
                'job_id': job_id,
                'status': 'completed',
                'input_info': video_info,
                'processing_time': processing_time,
                'total_time': total_time,
                'pipeline_result': result,
                'litmus_validation': validation_results,
                'passed': validation_results['all_criteria_met']
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {
                'video_source': video_source,
                'status': 'processing_failed',
                'error': str(e),
                'passed': False
            }
        finally:
            # Cleanup
            if os.path.exists(video_path) and video_path.startswith('/tmp'):
                os.remove(video_path)
            if os.path.exists(output_path):
                os.remove(output_path)
    
    def _prepare_video(self, video_source: str) -> str:
        """Download or prepare video for testing"""
        if video_source.startswith('http'):
            # Download video
            logger.info(f"Downloading video from {video_source}")
            video_path = f"/tmp/test_video_{int(time.time())}.mp4"
            
            response = requests.get(video_source, stream=True)
            response.raise_for_status()
            
            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return video_path
        
        elif os.path.exists(video_source):
            # Use local file
            return video_source
        
        else:
            # Create synthetic 45-minute test video
            logger.info("Creating synthetic 45-minute test video")
            return self._create_synthetic_long_video()
    
    def _create_synthetic_long_video(self) -> str:
        """Create a realistic 45-minute test video"""
        output_path = f"/tmp/synthetic_45min_{int(time.time())}.mp4"
        
        # Create 45-minute video with varying content
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', 'testsrc2=duration=2700:size=1920x1080:rate=30:decimals=2',
            '-f', 'lavfi',
            '-i', 'sine=frequency=440:duration=2700',
            '-filter_complex', 
            '''
            [0:v]scale=1920:1080,
            drawtext=text='Real World Test Video - %{pts\\:hms}':
            fontsize=48:fontcolor=white:x=50:y=50,
            drawtext=text='Simulated Dual Speaker Content':
            fontsize=32:fontcolor=yellow:x=50:y=150,
            drawtext=text='Face Region':
            fontsize=24:fontcolor=red:x=800:y=400:
            box=1:boxcolor=red@0.3:boxborderw=2[v];
            [v]drawbox=x=750:y=350:w=200:h=200:color=red@0.2:t=2[video]
            ''',
            '-map', '[video]',
            '-map', '1:a',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-t', '2700',  # 45 minutes
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Created synthetic video: {output_path}")
        
        return output_path
    
    def _validate_input_video(self, video_path: str) -> Dict[str, Any]:
        """Validate input video meets test criteria"""
        try:
            # Get video info with ffprobe
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration,size',
                '-show_entries', 'stream=width,height,codec_name',
                '-of', 'json', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            duration = float(info['format']['duration'])
            size_bytes = int(info['format']['size'])
            
            video_stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            
            # Validate criteria
            valid = True
            reasons = []
            
            # Duration: 40-50 minutes acceptable
            if duration < 2400 or duration > 3000:
                valid = False
                reasons.append(f"Duration {duration/60:.1f}min not in 40-50min range")
            
            # Resolution: at least 720p
            if width < 1280 or height < 720:
                valid = False
                reasons.append(f"Resolution {width}x{height} below 720p minimum")
            
            # File size: reasonable for 45 minutes
            if size_bytes > 5 * 1024**3:  # 5GB limit
                valid = False
                reasons.append(f"File size {size_bytes/1024**3:.1f}GB exceeds 5GB limit")
            
            return {
                'valid': valid,
                'reason': '; '.join(reasons) if reasons else 'Valid',
                'duration': duration,
                'resolution': f"{width}x{height}",
                'size_mb': size_bytes / (1024**2)
            }
            
        except Exception as e:
            return {
                'valid': False,
                'reason': f"Validation error: {e}",
                'duration': 0,
                'resolution': 'unknown',
                'size_mb': 0
            }
    
    def _validate_litmus_criteria(self, input_path: str, output_path: str, 
                                 job_id: str, processing_time: float) -> Dict[str, Any]:
        """Validate all 6 litmus test criteria"""
        results = {
            'ai_quality': False,
            'audio_quality': False,
            'visual_quality': False,
            'technical_compliance': False,
            'cost_efficiency': False,
            'performance': False,
            'details': {}
        }
        
        try:
            # 1. AI Quality: ≤ 5 highlights, 15-60s each
            highlights = self._validate_ai_quality(job_id)
            results['ai_quality'] = highlights['valid']
            results['details']['ai_quality'] = highlights
            
            # 2. Audio Quality: ≤ 1.5 LU spread
            audio_quality = self._validate_audio_quality(output_path)
            results['audio_quality'] = audio_quality['valid']
            results['details']['audio_quality'] = audio_quality
            
            # 3. Visual Quality: Face centered ≥ 90% of frames
            visual_quality = self._validate_visual_quality(output_path)
            results['visual_quality'] = visual_quality['valid']
            results['details']['visual_quality'] = visual_quality
            
            # 4. Technical Compliance: SDR, BT.709, level 4.0
            technical = self._validate_technical_compliance(output_path)
            results['technical_compliance'] = technical['valid']
            results['details']['technical_compliance'] = technical
            
            # 5. Cost Efficiency: < $4.00
            cost = self._validate_cost_efficiency(job_id)
            results['cost_efficiency'] = cost['valid']
            results['details']['cost_efficiency'] = cost
            
            # 6. Performance: ≤ 1.5x source duration
            input_duration = self._get_video_duration(input_path)
            ratio = processing_time / input_duration
            results['performance'] = ratio <= 1.5
            results['details']['performance'] = {
                'processing_time': processing_time,
                'source_duration': input_duration,
                'ratio': ratio,
                'valid': ratio <= 1.5
            }
            
            # Overall result
            results['all_criteria_met'] = all([
                results['ai_quality'],
                results['audio_quality'],
                results['visual_quality'],
                results['technical_compliance'],
                results['cost_efficiency'],
                results['performance']
            ])
            
        except Exception as e:
            logger.error(f"Litmus validation error: {e}")
            results['error'] = str(e)
        
        return results
    
    def _validate_ai_quality(self, job_id: str) -> Dict[str, Any]:
        """Validate AI-generated highlights"""
        highlights = self.db.find('highlight', {'job_id': job_id})
        
        valid_highlights = []
        for h in highlights:
            duration = h['end_time'] - h['start_time']
            if 15 <= duration <= 60:
                valid_highlights.append(h)
        
        return {
            'valid': len(valid_highlights) <= 5 and len(valid_highlights) > 0,
            'total_highlights': len(highlights),
            'valid_highlights': len(valid_highlights),
            'requirement': '≤ 5 highlights, 15-60s each'
        }
    
    def _validate_audio_quality(self, video_path: str) -> Dict[str, Any]:
        """Validate audio loudness spread with ebur128"""
        try:
            # Use ebur128 filter to measure loudness
            cmd = [
                'ffmpeg', '-i', video_path,
                '-af', 'ebur128=peak=true:framelog=verbose',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse integrated loudness from output
            lines = result.stderr.split('\n')
            loudness_measurements = []
            
            for line in lines:
                if 'I:' in line and 'LUFS' in line:
                    try:
                        # Extract loudness value
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'I:' and i + 1 < len(parts):
                                loudness = float(parts[i + 1])
                                loudness_measurements.append(loudness)
                                break
                    except (ValueError, IndexError):
                        continue
            
            if loudness_measurements:
                spread = max(loudness_measurements) - min(loudness_measurements)
                return {
                    'valid': spread <= 1.5,
                    'spread_lu': spread,
                    'measurements': len(loudness_measurements),
                    'requirement': '≤ 1.5 LU spread'
                }
            else:
                return {
                    'valid': False,
                    'spread_lu': float('inf'),
                    'error': 'No loudness measurements found',
                    'requirement': '≤ 1.5 LU spread'
                }
                
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'requirement': '≤ 1.5 LU spread'
            }
    
    def _validate_visual_quality(self, video_path: str) -> Dict[str, Any]:
        """Validate face centering using OpenCV"""
        try:
            cap = cv2.VideoCapture(video_path)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            total_frames = 0
            centered_frames = 0
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Define center third region
            center_x_min = frame_width // 3
            center_x_max = 2 * frame_width // 3
            center_y_min = frame_height // 3
            center_y_max = 2 * frame_height // 3
            
            # Sample every 30th frame (1 per second at 30fps)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % 30 != 0:
                    continue
                
                total_frames += 1
                
                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                
                # Check if any face is in center third
                for (x, y, w, h) in faces:
                    face_center_x = x + w // 2
                    face_center_y = y + h // 2
                    
                    if (center_x_min <= face_center_x <= center_x_max and
                        center_y_min <= face_center_y <= center_y_max):
                        centered_frames += 1
                        break
            
            cap.release()
            
            if total_frames == 0:
                return {
                    'valid': False,
                    'error': 'No frames processed',
                    'requirement': 'Face centered ≥ 90% of frames'
                }
            
            center_percentage = (centered_frames / total_frames) * 100
            
            return {
                'valid': center_percentage >= 90,
                'center_percentage': center_percentage,
                'centered_frames': centered_frames,
                'total_frames': total_frames,
                'requirement': 'Face centered ≥ 90% of frames'
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'requirement': 'Face centered ≥ 90% of frames'
            }
    
    def _validate_technical_compliance(self, video_path: str) -> Dict[str, Any]:
        """Validate technical specifications"""
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=color_space,color_primaries,profile,level',
                '-of', 'json', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            stream = info['streams'][0]
            color_primaries = stream.get('color_primaries', '')
            color_space = stream.get('color_space', '')
            profile = stream.get('profile', '')
            level = stream.get('level', '')
            
            # Check criteria
            is_bt709 = color_primaries == 'bt709'
            is_sdr = 'bt2020' not in color_space.lower()
            # Level 4.0 check (simplified)
            is_level_4 = '4' in str(level) if level else True
            
            return {
                'valid': is_bt709 and is_sdr and is_level_4,
                'color_primaries': color_primaries,
                'color_space': color_space,
                'profile': profile,
                'level': level,
                'is_bt709': is_bt709,
                'is_sdr': is_sdr,
                'is_level_4': is_level_4,
                'requirement': 'SDR, BT.709, level 4.0'
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'requirement': 'SDR, BT.709, level 4.0'
            }
    
    def _validate_cost_efficiency(self, job_id: str) -> Dict[str, Any]:
        """Validate cost under $4.00"""
        from budget_guard import budget_manager
        
        try:
            _, total_cost = budget_manager.check_budget(job_id)
            
            return {
                'valid': total_cost < 4.0,
                'total_cost': total_cost,
                'requirement': '< $4.00 total spend'
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'requirement': '< $4.00 total spend'
            }
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds"""
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    
    def _generate_summary(self):
        """Generate final validation summary"""
        total_videos = len(self.test_results['videos_tested'])
        passed_videos = sum(1 for v in self.test_results['videos_tested'] if v.get('passed', False))
        
        # Collect statistics
        processing_times = []
        costs = []
        
        for video in self.test_results['videos_tested']:
            if video.get('processing_time'):
                processing_times.append(video['processing_time'])
            
            cost_info = video.get('litmus_validation', {}).get('details', {}).get('cost_efficiency', {})
            if 'total_cost' in cost_info:
                costs.append(cost_info['total_cost'])
        
        self.test_results['summary'] = {
            'total_videos_tested': total_videos,
            'videos_passed': passed_videos,
            'success_rate': (passed_videos / total_videos * 100) if total_videos > 0 else 0,
            'average_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
            'average_cost': sum(costs) / len(costs) if costs else 0,
            'max_cost': max(costs) if costs else 0,
            'end_time': datetime.utcnow().isoformat()
        }


def main():
    """Main entry point for real-world validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-World Video Validation')
    parser.add_argument('--videos', nargs='+', help='Video sources (URLs or paths)')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic 45-min video')
    parser.add_argument('--count', type=int, default=3, help='Number of synthetic videos to test')
    
    args = parser.parse_args()
    
    validator = RealWorldValidator()
    
    # Determine video sources
    if args.videos:
        video_sources = args.videos
    elif args.synthetic:
        video_sources = ['synthetic'] * args.count
    else:
        # Default test with synthetic videos
        video_sources = ['synthetic'] * 3
    
    # Run validation
    logger.info(f"Testing {len(video_sources)} videos...")
    results = validator.run_real_world_validation(video_sources)
    
    # Save results
    timestamp = int(time.time())
    report_path = f"real_world_validation_report_{timestamp}.json"
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    summary = results['summary']
    logger.info("\n" + "=" * 60)
    logger.info("📊 REAL-WORLD VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Videos Tested: {summary['total_videos_tested']}")
    logger.info(f"Videos Passed: {summary['videos_passed']}")
    logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
    logger.info(f"Avg Processing Time: {summary['average_processing_time']:.1f}s")
    logger.info(f"Avg Cost: ${summary['average_cost']:.2f}")
    logger.info(f"Max Cost: ${summary['max_cost']:.2f}")
    
    if results['passed']:
        logger.info("\n✅ REAL-WORLD VALIDATION PASSED!")
        logger.info("System is ready for production deployment.")
    else:
        logger.error("\n❌ REAL-WORLD VALIDATION FAILED!")
        logger.error("Check individual video results for details.")
    
    logger.info(f"\nDetailed report saved to: {report_path}")
    
    return 0 if results['passed'] else 1


if __name__ == "__main__":
    exit(main())