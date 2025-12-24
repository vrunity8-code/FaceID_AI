import cv2
import numpy as np

def draw_results(img, faces, liveness_results, recognition_results, tracking_ids=None):
    """
    Draw bounding boxes, landmarks, and status text.
    """
    if tracking_ids is None:
        tracking_ids = [-1] * len(faces)

    for i, face in enumerate(faces):
        bbox = face.bbox.astype(int)
        track_id = tracking_ids[i]
        
        # Color based on recognition
        name = recognition_results[i]
        color = (0, 255, 0) if "Unknown" not in name and "Spoof" not in name else (0, 0, 255)
        
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Landmarks
        if face.kps is not None:
            for kp in face.kps:
                cv2.circle(img, (int(kp[0]), int(kp[1])), 2, (0, 0, 255), -1)

        # Liveness
        is_live_fas, score_fas = liveness_results[i]['fas']
        is_live_ppg, bpm, ppg_status = liveness_results[i]['ppg']
        
        status_color = (0, 255, 0) if is_live_fas else (0, 0, 255)
        status_text = f"FAS: {score_fas:.2f} {'REAL' if is_live_fas else 'SPOOF'}"
        
        cv2.putText(img, status_text, (bbox[0], bbox[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        cv2.putText(img, f"PPG: {ppg_status}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Recognition & ID
        id_text = f"ID: {track_id} | {name}"
        cv2.putText(img, id_text, (bbox[0], bbox[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img
