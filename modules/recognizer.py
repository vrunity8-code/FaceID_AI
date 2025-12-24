import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

class FaceRecognizer:
    def __init__(self):
        # Initialize FaceAnalysis with recognition
        # We use a separate instance or the same one, but for clarity here we separate.
        # In practice, FaceAnalysis(allowed_modules=['detection', 'recognition']) does both.
        # But to strictly follow "RetinaFace + ArcFace" separation request:
        # FaceAnalysis requires detection to be enabled even if we only want recognition
        self.app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, img, face):
        """
        Get 512-d embedding for a detected face.
        The 'face' object from detector already has kps, which recognizer uses for alignment.
        """
        try:
            # Ensure we have the recognition model
            if 'recognition' not in self.app.models:
                print("Error: Recognition model not loaded.")
                return np.zeros(512)

            model = self.app.models['recognition']
            
            # The model.get(img, face) method aligns the face using face.kps and returns the embedding
            # It modifies the 'face' object in-place (adds .embedding) or returns it?
            # Actually, InsightFace models usually modify the face object or return the embedding.
            # Let's check typical usage. Usually: feat = model.get(img, face) returns the embedding directly 
            # OR it attaches it to face.embedding.
            # In recent versions, it attaches to face.embedding.
            
            model.get(img, face)
            
            if face.embedding is not None:
                return face.embedding
            else:
                print("Warning: No embedding generated.")
                return np.zeros(512)
                
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return np.zeros(512)

    def compute_similarity(self, embed1, embed2):
        return np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
