"""
Test script for 3D Face Alignment
Verifies pose estimation and mesh reconstruction
"""

import cv2
import numpy as np
from modules.detector import FaceDetector
from modules.alignment_3d import FaceAlignment3D
from modules.utils_3d import draw_face_mesh_overlay, draw_pose_info


def test_3d_alignment():
    """Test 3D alignment on webcam feed"""
    print("=" * 50)
    print("3D Face Alignment Test")
    print("=" * 50)
    
    # Initialize modules
    print("\n1. Initializing detector...")
    detector = FaceDetector()
    print("   [OK] Detector initialized")
    
    print("\n2. Initializing 3D alignment...")
    align_3d = FaceAlignment3D()
    print("   [OK] 3D Alignment initialized")
    
    # Open webcam
    print("\n3. Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("   [ERROR] Could not open camera")
        return False
    print("   [OK] Webcam opened")
    
    print("\n4. Starting test...")
    print("   Controls:")
    print("      - Press 'q' to exit")
    print("      - Press 't' to toggle mesh")
    print("      - Press 'p' to toggle pose axes")
    print("      - Press 'l' to toggle landmarks")
    
    show_mesh = True
    show_axes = True
    show_landmarks = True
    
    frame_count = 0
    success_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect faces
        faces = detector.detect(frame)
        
        # Process each face
        for face in faces:
            # Estimate pose
            pose_info = align_3d.estimate_pose(frame, face)
            
            if pose_info is not None:
                success_count += 1
                
                # Reconstruct mesh
                mesh_data = align_3d.reconstruct_mesh(frame, face, pose_info)
                
                # Draw visualization
                frame = draw_face_mesh_overlay(
                    frame, face, mesh_data, pose_info,
                    show_mesh=show_mesh,
                    show_landmarks=show_landmarks,
                    show_axes=show_axes,
                    show_pose_text=True
                )
        
        # Show statistics
        if frame_count > 0:
            success_rate = (success_count / frame_count) * 100
            cv2.putText(frame, f"Success Rate: {success_rate:.1f}%", 
                       (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('3D Alignment Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            show_mesh = not show_mesh
            print(f"   Mesh: {'ON' if show_mesh else 'OFF'}")
        elif key == ord('p'):
            show_axes = not show_axes
            print(f"   Pose axes: {'ON' if show_axes else 'OFF'}")
        elif key == ord('l'):
            show_landmarks = not show_landmarks
            print(f"   Landmarks: {'ON' if show_landmarks else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print results
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"  Total frames: {frame_count}")
    print(f"  Successful alignments: {success_count}")
    if frame_count > 0:
        print(f"  Success rate: {(success_count/frame_count)*100:.1f}%")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    try:
        test_3d_alignment()
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()

