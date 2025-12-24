import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self, det_size=(640, 640)):
        # Initialize FaceAnalysis with detection only
        self.app = FaceAnalysis(allowed_modules=['detection'])
        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect(self, img):
        """
        Detect faces in the image.
        Returns a list of Face objects (bbox, kps, etc.)
        """
        faces = self.app.get(img)
        return faces

    def align_face(self, img, face):
        """
        Align face using 5 landmarks.
        InsightFace's FaceAnalysis handles alignment internally when extracting features,
        but we can explicitly do it if needed for other models.
        Here we just return the crop for visualization or external use if needed.
        """
        # Simple crop based on bbox for now, or use face.embedding if using full pipeline
        bbox = face.bbox.astype(int)
        return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
