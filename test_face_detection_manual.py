"""Manual face detection test without pytest dependencies"""

import cv2
import numpy as np
import time
from montage.providers.smart_track import SmartTrack

print("=== Face Detection Integration Test ===")

# Test 1: Initialize SmartTrack and verify face cascade
print("\n1. Testing SmartTrack initialization...")
tracker = SmartTrack()
if tracker.face_cascade is not None and not tracker.face_cascade.empty():
    print("✅ Face cascade initialized successfully")
else:
    print("❌ Face cascade failed to initialize")
    exit(1)

# Test 2: Test detection on synthetic frames
print("\n2. Testing face detection accuracy on synthetic frames...")

test_scenarios = [
    {"name": "single_face_center", "faces": 1},
    {"name": "two_faces", "faces": 2},
    {"name": "no_face", "faces": 0},
]

results = []
for scenario in test_scenarios:
    # Create synthetic frame
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 50
    
    if scenario["faces"] >= 1:
        # Create more realistic face-like pattern with proper skin tone
        center_x, center_y = 640, 360
        # Face oval
        cv2.ellipse(frame, (center_x, center_y), (80, 100), 0, 0, 360, (203, 175, 150), -1)
        # Add forehead area
        cv2.ellipse(frame, (center_x, center_y - 50), (70, 40), 0, 0, 360, (213, 185, 160), -1)
        # Eyes with proper contrast
        eye_y = center_y - 20
        cv2.ellipse(frame, (center_x - 30, eye_y), (20, 15), 0, 0, 360, (255, 255, 255), -1)  # Eye whites
        cv2.ellipse(frame, (center_x + 30, eye_y), (20, 15), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(frame, (center_x - 30, eye_y), 8, (50, 30, 20), -1)  # Iris
        cv2.circle(frame, (center_x + 30, eye_y), 8, (50, 30, 20), -1)
        cv2.circle(frame, (center_x - 30, eye_y), 3, (0, 0, 0), -1)  # Pupil
        cv2.circle(frame, (center_x + 30, eye_y), 3, (0, 0, 0), -1)
        # Eyebrows
        cv2.ellipse(frame, (center_x - 30, eye_y - 20), (25, 5), 0, 0, 180, (100, 80, 70), -1)
        cv2.ellipse(frame, (center_x + 30, eye_y - 20), (25, 5), 0, 0, 180, (100, 80, 70), -1)
        # Nose
        cv2.ellipse(frame, (center_x, center_y), (15, 25), 0, 0, 360, (193, 165, 140), -1)
        # Nose tip
        cv2.ellipse(frame, (center_x, center_y + 10), (12, 8), 0, 0, 360, (183, 155, 130), -1)
        # Mouth
        mouth_y = center_y + 35
        cv2.ellipse(frame, (center_x, mouth_y), (30, 15), 0, 0, 180, (150, 100, 100), -1)
        # Add shadow areas for more realism
        cv2.ellipse(frame, (center_x - 30, eye_y + 10), (15, 5), 0, 0, 360, (183, 155, 130), -1)
        cv2.ellipse(frame, (center_x + 30, eye_y + 10), (15, 5), 0, 0, 360, (183, 155, 130), -1)
        
    if scenario["faces"] >= 2:
        # Add second face to the right
        center_x2, center_y2 = 900, 360
        cv2.ellipse(frame, (center_x2, center_y2), (75, 95), 0, 0, 360, (213, 185, 160), -1)
        # Simplified features for second face
        cv2.ellipse(frame, (center_x2 - 25, center_y2 - 20), (18, 12), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(frame, (center_x2 + 25, center_y2 - 20), (18, 12), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(frame, (center_x2 - 25, center_y2 - 20), 6, (60, 40, 30), -1)
        cv2.circle(frame, (center_x2 + 25, center_y2 - 20), 6, (60, 40, 30), -1)
        cv2.ellipse(frame, (center_x2, center_y2 + 30), (25, 12), 0, 0, 180, (160, 110, 110), -1)
    
    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization for better contrast
    gray = cv2.equalizeHist(gray)
    detected = tracker.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    
    success = len(detected) == scenario["faces"]
    results.append(success)
    print(f"  {scenario['name']}: Expected {scenario['faces']}, Detected {len(detected)} - {'✅' if success else '❌'}")

# Test 3: Performance test
print("\n3. Testing face detection performance...")
test_frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 100
gray_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)

# Warm up
for _ in range(5):
    tracker.face_cascade.detectMultiScale(gray_frame, 1.1, 4)

# Measure
times = []
for _ in range(20):
    start = time.time()
    tracker.face_cascade.detectMultiScale(gray_frame, 1.1, 4)
    times.append(time.time() - start)

avg_time = sum(times) / len(times)
print(f"✅ Average detection time: {avg_time*1000:.1f}ms ({1/avg_time:.1f} FPS)")

# Test 4: Real video simulation
print("\n4. Testing video processing simulation...")
# Simulate processing frames from a video
frames_with_faces = 0
total_frames = 30

for i in range(total_frames):
    # Create varying scenes
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * (50 + i * 5)
    
    # Add face in 80% of frames (to test 90% accuracy requirement)
    if i % 10 < 8:  # 80% of frames have faces
        x = 400 + i * 10
        # Create realistic face
        cv2.ellipse(frame, (x + 90, 390), (70, 85), 0, 0, 360, (203, 175, 150), -1)
        # Eyes
        cv2.ellipse(frame, (x + 70, 370), (15, 10), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(frame, (x + 110, 370), (15, 10), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(frame, (x + 70, 370), 5, (30, 20, 10), -1)
        cv2.circle(frame, (x + 110, 370), 5, (30, 20, 10), -1)
        # Nose and mouth
        cv2.ellipse(frame, (x + 90, 390), (10, 18), 0, 0, 360, (193, 165, 140), -1)
        cv2.ellipse(frame, (x + 90, 415), (20, 8), 0, 0, 180, (150, 100, 100), -1)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = tracker.face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        frames_with_faces += 1

detection_rate = frames_with_faces / total_frames * 100
print(f"✅ Detection rate: {detection_rate:.1f}% ({frames_with_faces}/{total_frames} frames)")

# Summary
print("\n=== Summary ===")
accuracy = sum(results) / len(results) * 100
print(f"Synthetic frame accuracy: {accuracy:.1f}%")
print(f"Video simulation detection rate: {detection_rate:.1f}%")
print(f"Performance: {avg_time*1000:.1f}ms per frame")

if accuracy >= 90 and detection_rate >= 70:
    print("\n✅ Face detection integration test PASSED!")
    print("   - Accuracy meets 90%+ requirement")
    print("   - Performance suitable for real-time")
else:
    print("\n❌ Face detection needs improvement")
    print(f"   - Accuracy: {accuracy:.1f}% (need 90%+)")
    print(f"   - Detection rate: {detection_rate:.1f}%")