import cv2
import numpy as np
import time
import pickle
import os
from modules.detector import FaceDetector
from modules.recognizer import FaceRecognizer
from modules.liveness.fas import FASDetector
from modules.liveness.ppg import PPGLiveness
from modules.tracker import SimpleTracker
from modules.utils import draw_results
from modules.alignment_3d import FaceAlignment3D
from modules.utils_3d import draw_face_mesh_overlay, draw_pose_info

# Database file
DB_FILE = 'faces.pkl'

def load_faces():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading database: {e}")
    return {}

def save_faces(database):
    try:
        with open(DB_FILE, 'wb') as f:
            pickle.dump(database, f)
        print("Database saved.")
    except Exception as e:
        print(f"Error saving database: {e}")

def main():
    # Initialize modules
    print("Initializing modules...")
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    fas = FASDetector(model_path='models/fas.onnx')
    tracker = SimpleTracker()
    align_3d = FaceAlignment3D()  # Initialize 3D alignment
    print("3D Alignment initialized.")
    
    # State management
    ppg_instances = {} # face_id -> PPGLiveness instance
    known_embeddings = load_faces()
    print(f"Loaded {len(known_embeddings)} registered faces.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting video loop.")
    print("Controls:")
    print("  'q': Exit")
    print("  'r': Register the current face (if only one face visible)")
    print("  't': Toggle 3D mesh visualization")
    print("  'p': Toggle pose axes display")
    print("  'l': Toggle 3D landmarks")
    
    register_mode = False
    register_name = ""
    
    # 3D visualization toggles
    show_3d_mesh = True
    show_pose_axes = True
    show_3d_landmarks = True
    
    # Config
    SKIP_FRAMES = 5 # Run recognition every N frames
    frame_count = 0
    
    # State
    register_mode = False
    register_name = ""
    
    # Cache results: track_id -> (name, liveness_info)
    # We will update this cache periodically
    results_cache = {} 

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 1. Detection & Tracking (Every frame for smoothness)
        faces = detector.detect(frame)
        
        rects = []
        for face in faces:
            rects.append(face.bbox.astype(int).tolist())
        
        tracked_objects = tracker.update(rects)
        
        # Map tracked objects back to faces
        faces_with_ids = []
        current_track_ids = []
        
        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            match_id = -1
            for tid, trect in tracked_objects:
                if trect == bbox:
                    match_id = tid
                    break
            faces_with_ids.append((face, match_id))
            current_track_ids.append(match_id)

        # 2. Recognition & Liveness (Interval based or New ID)
        # We run if frame_count % SKIP_FRAMES == 0 OR if we see a new ID not in cache
        should_run_heavy = (frame_count % SKIP_FRAMES == 0)
        
        # Check for new IDs
        for tid in current_track_ids:
            if tid != -1 and tid not in results_cache:
                should_run_heavy = True
                break
        
        if should_run_heavy:
            for face, face_id in faces_with_ids:
                # Recognition
                identity = "Unknown"
                max_score = 0
                
                emb = recognizer.get_embedding(frame, face)
                if np.any(emb):
                    for name, known_emb in known_embeddings.items():
                        score = recognizer.compute_similarity(emb, known_emb)
                        if score > max_score:
                            max_score = score
                            identity = name
                    
                    if max_score > 0.4:
                        identity = f"{identity} ({max_score:.2f})"
                    else:
                        identity = "Unknown"
                
                # FAS Liveness
                fas_score, is_live_fas = fas.check_liveness(frame, face.bbox)
                if not is_live_fas:
                    identity = f"SPOOF {identity}"
                
                # Update cache
                if face_id != -1:
                    # Keep existing PPG state if present
                    ppg_state = results_cache.get(face_id, {}).get('ppg', None)
                    results_cache[face_id] = {
                        'name': identity,
                        'fas': (is_live_fas, fas_score),
                        'ppg': ppg_state # Preserve PPG state
                    }

        # 3. PPG Liveness (Every frame, but lightweight-ish)
        # We need continuous frames for PPG, so we run update() every frame
        for face, face_id in faces_with_ids:
            if face_id != -1:
                if face_id not in ppg_instances:
                    ppg_instances[face_id] = PPGLiveness()
                
                is_live_ppg, bpm, ppg_status = ppg_instances[face_id].update(frame, face.bbox)
                
                # Update cache with new PPG result
                if face_id in results_cache:
                    results_cache[face_id]['ppg'] = (is_live_ppg, bpm, ppg_status)
                else:
                     # If not in cache yet (e.g. first frame of new face), init cache
                     results_cache[face_id] = {
                        'name': "Analyzing...",
                        'fas': (True, 0.0), # Assume real until checked
                        'ppg': (is_live_ppg, bpm, ppg_status)
                    }

        # Cleanup cache
        active_ids_set = set(current_track_ids)
        for fid in list(results_cache.keys()):
            if fid not in active_ids_set:
                del results_cache[fid]
        for fid in list(ppg_instances.keys()):
            if fid not in active_ids_set:
                del ppg_instances[fid]

        # Prepare results for drawing
        liveness_results_list = []
        recognition_results_list = []
        
        for face, face_id in faces_with_ids:
            if face_id != -1 and face_id in results_cache:
                res = results_cache[face_id]
                name = res.get('name', 'Unknown')
                fas_res = res.get('fas', (True, 0.0))
                ppg_res = res.get('ppg', (False, 0, "Init"))
                if ppg_res is None: ppg_res = (False, 0, "Init")
                
                recognition_results_list.append(name)
                liveness_results_list.append({'fas': fas_res, 'ppg': ppg_res})
            else:
                # Fallback for untracked or not-yet-processed
                recognition_results_list.append("Analyzing...")
                liveness_results_list.append({'fas': (True, 0.0), 'ppg': (False, 0, "Init")})

        # 4. 3D Alignment and Pose Estimation
        # Process 3D alignment for each face
        pose_data = {}  # face_id -> pose_info
        mesh_data = {}  # face_id -> mesh_data
        
        for face, face_id in faces_with_ids:
            if face_id != -1:
                # Estimate pose
                pose_info = align_3d.estimate_pose(frame, face)
                if pose_info is not None:
                    pose_data[face_id] = pose_info
                    
                    # Reconstruct mesh
                    mesh = align_3d.reconstruct_mesh(frame, face, pose_info)
                    if mesh is not None:
                        mesh_data[face_id] = mesh
        
        # Draw base results
        frame = draw_results(frame, faces, liveness_results_list, recognition_results_list, current_track_ids)
        
        # Draw 3D overlays
        for face, face_id in faces_with_ids:
            if face_id != -1:
                pose_info = pose_data.get(face_id)
                mesh = mesh_data.get(face_id)
                
                if show_3d_mesh or show_pose_axes or show_3d_landmarks:
                    frame = draw_face_mesh_overlay(
                        frame, face, mesh, pose_info,
                        show_mesh=show_3d_mesh,
                        show_landmarks=show_3d_landmarks,
                        show_axes=show_pose_axes,
                        show_pose_text=False  # We show it separately
                    )
                
                # Draw pose info on top-left corner for first face
                if pose_info is not None and face_id == current_track_ids[0]:
                    frame = draw_pose_info(frame, pose_info, position=(10, 30))

        # Registration UI
        if register_mode:
            # Dark overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            cv2.putText(frame, "REGISTRATION MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Name: {register_name}_", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, "Enter: Save | Esc: Cancel", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            # Show controls and 3D status
            status_y = frame.shape[0] - 50
            cv2.putText(frame, "'r': Register | 'q': Quit | 't': 3D Mesh | 'p': Pose Axes | 'l': Landmarks", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show 3D visualization status
            status_text = f"3D: Mesh[{'ON' if show_3d_mesh else 'OFF'}] Pose[{'ON' if show_pose_axes else 'OFF'}] Landmarks[{'ON' if show_3d_landmarks else 'OFF'}]"
            cv2.putText(frame, status_text, (10, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow('FaceID System', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and not register_mode:
            if len(faces) == 1:
                register_mode = True
                register_name = ""
            else:
                print("Registration requires exactly one face.")
        elif key == ord('t'):  # Toggle 3D mesh
            show_3d_mesh = not show_3d_mesh
            print(f"3D Mesh: {'ON' if show_3d_mesh else 'OFF'}")
        elif key == ord('p'):  # Toggle pose axes
            show_pose_axes = not show_pose_axes
            print(f"Pose Axes: {'ON' if show_pose_axes else 'OFF'}")
        elif key == ord('l'):  # Toggle 3D landmarks
            show_3d_landmarks = not show_3d_landmarks
            print(f"3D Landmarks: {'ON' if show_3d_landmarks else 'OFF'}")
        
        if register_mode:
            if key == 13: # Enter
                if len(faces) == 1 and len(register_name) > 0:
                    face = faces[0]
                    emb = recognizer.get_embedding(frame, face)
                    known_embeddings[register_name] = emb
                    save_faces(known_embeddings)
                    print(f"Registered {register_name}")
                    register_mode = False
                    # Force update cache
                    results_cache = {} 
                elif len(faces) != 1:
                    print("Registration failed: Face lost or multiple faces.")
            elif key == 27: # Esc
                register_mode = False
            elif key == 8: # Backspace
                register_name = register_name[:-1]
            elif 32 <= key <= 126:
                register_name += chr(key)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
