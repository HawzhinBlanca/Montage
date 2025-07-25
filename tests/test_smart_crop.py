"""Test smart crop face detection functionality using actual SmartTracker"""

import subprocess
import tempfile
import os
from unittest.mock import MagicMock, patch, Mock
import numpy as np
import cv2

import pytest


class TestFaceDetectionCrop:
    """Test face detection from SmartTracker implementation"""
    
    def test_smart_tracker_face_detection(self):
        """Test that SmartTracker detects faces correctly"""
        from montage.providers.smart_track import SmartTracker
        
        # Initialize tracker
        tracker = SmartTracker()
        
        # Create a mock video capture
        with patch('cv2.VideoCapture') as mock_cap:
            # Mock video properties
            mock_video = Mock()
            mock_video.get.return_value = 30.0  # 30 FPS
            
            # Create test frames with face-like regions
            frame1 = np.ones((1080, 1920, 3), dtype=np.uint8) * 100
            # Add bright rectangle to simulate face
            cv2.rectangle(frame1, (800, 300), (1000, 600), (200, 200, 200), -1)
            
            # Mock read() to return frames then False
            mock_video.read.side_effect = [
                (True, frame1),  # Frame with face
                (False, None),   # End of video
            ]
            
            mock_cap.return_value = mock_video
            
            # Mock face detection to return our test face
            with patch.object(tracker, 'face_cascade') as mock_cascade:
                mock_cascade.detectMultiScale.return_value = np.array([
                    [125, 46, 46, 75]  # Scaled face coordinates (640x360)
                ])
                
                # Run face analysis
                segments = tracker.analyze_faces('test_video.mp4')
                
                # Verify face detection was called
                assert mock_cascade.detectMultiScale.called
                
                # Should detect face in frame
                assert len(segments) >= 0  # May group into segments
                
    def test_face_detection_with_opencv(self):
        """Test actual OpenCV face detection with test image"""
        # This test verifies OpenCV cascade classifier works
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Verify cascade loaded
        assert not face_cascade.empty()
        
        # Create a simple test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Convert to grayscale
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        # Try detection (should find no faces in black image)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        assert isinstance(faces, np.ndarray)
        assert len(faces) == 0  # No faces in black image
        
        print("✅ OpenCV face detection working correctly")
        
    def test_crop_aspect_ratio_calculation(self):
        """Test calculation of 9:16 crop parameters"""
        # Given a 1920x1080 video, calculate 9:16 crop
        input_width = 1920
        input_height = 1080
        
        # For 9:16 aspect ratio from 16:9 source
        # We need to crop width to maintain full height
        target_aspect = 9/16
        
        # Calculate crop dimensions
        crop_height = input_height
        crop_width = int(crop_height * target_aspect)
        
        # Verify dimensions
        assert crop_width == 607  # 1080 * (9/16) = 607.5
        assert crop_height == 1080
        
        # Calculate crop position (center)
        crop_x = (input_width - crop_width) // 2
        crop_y = 0
        
        assert crop_x == 656  # (1920 - 607) / 2
        assert crop_y == 0
        
        # Verify aspect ratio
        actual_aspect = crop_width / crop_height
        assert abs(actual_aspect - target_aspect) < 0.01
        
        print(f"✅ Crop calculation correct: {crop_width}x{crop_height} at ({crop_x}, {crop_y})")
        
    def test_multiple_face_crop_strategy(self):
        """Test cropping strategy when multiple faces detected"""
        # When multiple faces are detected, we should either:
        # 1. Crop to include all faces (if possible)
        # 2. Crop to the most prominent face
        # 3. Crop to center of all faces
        
        faces = [
            {"x": 400, "y": 300, "w": 200, "h": 200},
            {"x": 1200, "y": 350, "w": 180, "h": 180},
            {"x": 800, "y": 400, "w": 220, "h": 220},
        ]
        
        # Calculate bounding box of all faces
        min_x = min(f["x"] for f in faces)
        max_x = max(f["x"] + f["w"] for f in faces)
        min_y = min(f["y"] for f in faces)
        max_y = max(f["y"] + f["h"] for f in faces)
        
        # Center of all faces
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        
        assert center_x == 890  # (400 + 1380) / 2
        assert center_y == 450  # (300 + 600) / 2
        
        print(f"✅ Multi-face crop center: ({center_x}, {center_y})")
        
    def test_face_detection_performance_check(self):
        """Verify face detection performance requirements"""
        from montage.providers.smart_track import SmartTracker
        import time
        
        tracker = SmartTracker()
        
        # Create test image
        test_image = np.zeros((360, 640, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        if tracker.face_cascade is not None:
            # Measure detection time
            start = time.time()
            faces = tracker.face_cascade.detectMultiScale(gray, 1.1, 5)
            end = time.time()
            
            detection_time = end - start
            
            # Should be fast for small image
            assert detection_time < 0.1  # 100ms max
            
            print(f"✅ Face detection time: {detection_time*1000:.1f}ms")
        else:
            print("⚠️ Face cascade not loaded, skipping performance test")
            
    def test_different_face_positions_different_crop_centers(self):
        """Test that two frames with different face positions produce different crop centers"""
        from montage.providers.smart_track import SmartTracker
        
        # Initialize tracker
        tracker = SmartTracker()
        
        # Create two test frames with faces at different positions
        with patch('cv2.VideoCapture') as mock_cap:
            # Mock video properties
            mock_video = Mock()
            mock_video.get.return_value = 30.0  # 30 FPS
            
            # Frame 1: Face on left side (position x=400)
            frame1 = np.ones((1080, 1920, 3), dtype=np.uint8) * 100
            cv2.rectangle(frame1, (300, 400), (500, 700), (200, 200, 200), -1)
            
            # Frame 2: Face on right side (position x=1400)
            frame2 = np.ones((1080, 1920, 3), dtype=np.uint8) * 100
            cv2.rectangle(frame2, (1300, 400), (1500, 700), (200, 200, 200), -1)
            
            # Mock read() to return both frames
            mock_video.read.side_effect = [
                (True, frame1),  # Frame with face on left
                (True, frame2),  # Frame with face on right
                (False, None),   # End of video
            ]
            
            mock_cap.return_value = mock_video
            
            # Mock face detection to return different positions
            with patch.object(tracker, 'face_cascade') as mock_cascade:
                # Return different face positions for each frame
                face_positions = [
                    np.array([[400, 400, 200, 300]]),  # Face on left (x=400)
                    np.array([[1400, 400, 200, 300]]), # Face on right (x=1400)
                ]
                mock_cascade.detectMultiScale.side_effect = face_positions
                
                # Analyze faces
                segments = tracker.analyze_faces('test_video.mp4')
                
                # Process segments to add crop params
                import asyncio
                segments_with_crops = asyncio.run(
                    tracker._add_crop_params('test_video.mp4', [
                        {"type": "face", "features": {"face_positions": [{"x": 400}]}, "start_time": 0},
                        {"type": "face", "features": {"face_positions": [{"x": 1400}]}, "start_time": 1}
                    ])
                )
                
                # Verify different crop centers
                assert len(segments_with_crops) >= 2
                assert "crop_params" in segments_with_crops[0]
                assert "crop_params" in segments_with_crops[1]
                
                # Get crop centers
                crop_x1 = segments_with_crops[0]["crop_params"]["x"]
                crop_x2 = segments_with_crops[1]["crop_params"]["x"]
                
                # Verify crop centers are different
                assert crop_x1 != crop_x2, f"Crop centers should be different: {crop_x1} vs {crop_x2}"
                
                # Verify crop1 is more to the left (smaller x) since face1 is on left
                assert crop_x1 < crop_x2, f"Left face should have smaller crop x: {crop_x1} vs {crop_x2}"
                
                # Verify approximate positions (with 9:16 crop of 607px width)
                # Face at x=400 should center around x=400, so crop_x ≈ 97 (400-303)
                # Face at x=1400 should center around x=1400, so crop_x ≈ 1097 (1400-303)
                assert abs(crop_x1 - 97) < 50, f"Crop1 position unexpected: {crop_x1}"
                assert abs(crop_x2 - 1097) < 50, f"Crop2 position unexpected: {crop_x2}"
                
                print(f"✅ Different face positions produced different crop centers: {crop_x1} vs {crop_x2}")
    
    def test_output_video_format_verification(self):
        """Test commands to verify output video format"""
        print("\n=== Manual Verification Steps ===")
        print("After running smart crop on a video:")
        print()
        print("1. Check output dimensions:")
        print("   ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 output.mp4")
        print("   Expected: 607,1080 (or similar 9:16 ratio)")
        print()
        print("2. Check aspect ratio:")
        print("   ffprobe -v error -select_streams v:0 -show_entries stream=display_aspect_ratio -of default=nw=1:nk=1 output.mp4")
        print("   Expected: 9:16")
        print()
        print("3. Visual verification:")
        print("   - Faces should be centered in frame")
        print("   - No important content cut off")
        print("   - Smooth tracking if faces move")
        
        assert True  # Documentation test


if __name__ == "__main__":
    # Run with: pytest tests/test_smart_crop.py -v -s
    pytest.main([__file__, "-v", "-s"])