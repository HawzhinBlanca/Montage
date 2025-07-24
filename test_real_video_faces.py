"""Test face detection on real video"""
import cv2
import sys

def test_real_video_faces(video_path):
    print(f"Testing face detection on: {video_path}")
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("❌ Failed to load face cascade")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video info: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s duration")
    
    face_frames = 0
    total_faces = 0
    frames_checked = 0
    
    # Sample every 30 frames (roughly 1 per second)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 30 == 0:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Try different preprocessing
            # Option 1: Histogram equalization
            gray_eq = cv2.equalizeHist(gray)
            
            # Detect faces with different parameters
            faces1 = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            faces2 = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            faces3 = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
            
            # Take the detection with most faces
            faces = faces1
            if len(faces2) > len(faces):
                faces = faces2
            if len(faces3) > len(faces):
                faces = faces3
            
            if len(faces) > 0:
                face_frames += 1
                total_faces += len(faces)
                timestamp = frame_count / fps
                print(f"  Frame {frame_count} ({timestamp:.1f}s): {len(faces)} face(s) detected")
            
            frames_checked += 1
        
        frame_count += 1
        
        # Progress update every 5 seconds
        if frame_count % (int(fps) * 5) == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}%")
    
    cap.release()
    
    # Results
    print("\n=== Results ===")
    print(f"Frames checked: {frames_checked}")
    print(f"Frames with faces: {face_frames} ({face_frames/frames_checked*100:.1f}%)")
    print(f"Total faces detected: {total_faces}")
    print(f"Average faces per frame with faces: {total_faces/face_frames:.1f}" if face_frames > 0 else "N/A")
    
    # Check if this meets requirements
    detection_rate = face_frames / frames_checked * 100
    if detection_rate >= 90:
        print(f"\n✅ Detection rate {detection_rate:.1f}% meets 90%+ requirement!")
    else:
        print(f"\n⚠️ Detection rate {detection_rate:.1f}% below 90% requirement")

if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/hawzhin/Montage/test_video_5min.mp4"
    test_real_video_faces(video_path)