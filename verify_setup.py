import sys
import os
import cv2
import numpy as np

print("Verifying imports...")
try:
    import insightface
    import onnxruntime
    from modules.detector import FaceDetector
    from modules.recognizer import FaceRecognizer
    from modules.liveness.fas import FASDetector
    from modules.liveness.ppg import PPGLiveness
    from modules.tracker import SimpleTracker
    print("Imports successful.")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

print("Verifying initialization...")
try:
    print("Initializing Detector...")
    detector = FaceDetector()
    print("Initializing Recognizer...")
    recognizer = FaceRecognizer()
    print("Initializing FAS...")
    fas = FASDetector(model_path='models/fas.onnx')
    print("Initializing Tracker...")
    tracker = SimpleTracker()
    print("Initializing PPG...")
    ppg = PPGLiveness()
    print("Initialization successful.")
except Exception as e:
    print(f"Initialization failed: {e}")
    sys.exit(1)

print("System is ready to run.")
