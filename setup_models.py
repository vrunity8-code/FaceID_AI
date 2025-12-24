import os
import urllib.request
import insightface
from insightface.app import FaceAnalysis

def download_fas_model():
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_path = 'models/fas.onnx'
    # Check if exists and size is reasonable (>1MB)
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        if size > 1000000: # 1MB
            print(f"FAS model already exists at {model_path} ({size} bytes)")
            return
        else:
            print(f"FAS model exists but is too small ({size} bytes). Re-downloading...")
            os.remove(model_path)

    print("Downloading FAS model (MiniFASNetV2)...")
    # Try original repo URL
    url = "https://github.com/kprokofi/light-weight-face-anti-spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.onnx"
    try:
        urllib.request.urlretrieve(url, model_path)
        print("FAS model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading FAS model: {e}")
        print("Please manually download '2.7_80x80_MiniFASNetV2.onnx' and save it as 'models/fas.onnx'")

def download_insightface_models():
    print("Triggering InsightFace model download...")
    try:
        app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("InsightFace models downloaded successfully.")
    except Exception as e:
        print(f"Error downloading InsightFace models: {e}")

if __name__ == "__main__":
    download_fas_model()
    download_insightface_models()
