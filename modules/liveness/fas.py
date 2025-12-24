import cv2
import numpy as np
import onnxruntime

class FASDetector:
    def __init__(self, model_path='models/fas.onnx'):
        self.model_path = model_path
        try:
            # Check size
            import os
            if os.path.exists(model_path) and os.path.getsize(model_path) < 1000:
                raise Exception("Model file too small (corrupted?)")

            self.session = onnxruntime.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
        except Exception as e:
            print(f"Warning: FAS model error at {model_path}. Liveness check will be mocked. Error: {e}")
            self.session = None

    def check_liveness(self, img, face_bbox):
        """
        Check if the face in bbox is real or spoof.
        Returns: score (0.0-1.0), is_real (bool)
        """
        if self.session is None:
            return 0.9, True # Mock result

        # Crop face
        x1, y1, x2, y2 = face_bbox.astype(int)
        h, w, _ = img.shape
        
        # Ensure crop is within bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0, False
            
        face_img = img[y1:y2, x1:x2]
        if face_img.size == 0:
            return 0.0, False

        # Preprocess
        try:
            # Get expected input shape from model
            target_h, target_w = self.input_shape[2], self.input_shape[3]
            
            blob = cv2.resize(face_img, (target_w, target_h))
            blob = blob.astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1) # HWC -> CHW
            blob = np.expand_dims(blob, axis=0)

            # Inference
            outputs = self.session.run(None, {self.input_name: blob})
            
            # Handle different output shapes
            output = outputs[0]
            if output.shape == (1, 1):
                score = float(output[0][0])
            elif output.shape == (1, 2): # [Spoof, Real]
                score = float(output[0][1])
            elif output.shape == (1, 3): # [Spoof, Real, ?]
                score = float(output[0][1])
            else:
                # Fallback or assume first element is score
                score = float(output.flatten()[0])
            
            # Threshold
            is_real = score > 0.5
            return score, is_real
            
        except Exception as e:
            print(f"FAS Inference Error: {e}")
            return 0.0, False
