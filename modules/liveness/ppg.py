import numpy as np
import cv2
from scipy import signal

class PPGLiveness:
    def __init__(self, buffer_size=150, fps=30):
        self.buffer_size = buffer_size
        self.fps = fps
        self.signal_buffer = []
        self.times = []

    def update(self, img, face_bbox):
        """
        Update PPG signal buffer with new frame.
        Returns: is_live (bool), bpm (float), status (str)
        """
        # Extract ROI (forehead or cheeks)
        x1, y1, x2, y2 = face_bbox.astype(int)
        
        # Simple ROI: Center of the face
        w = x2 - x1
        h = y2 - y1
        roi_x1 = x1 + int(w * 0.3)
        roi_x2 = x1 + int(w * 0.7)
        roi_y1 = y1 + int(h * 0.1) # Forehead area roughly
        roi_y2 = y1 + int(h * 0.3)
        
        # Check bounds
        img_h, img_w, _ = img.shape
        roi_x1 = max(0, roi_x1)
        roi_y1 = max(0, roi_y1)
        roi_x2 = min(img_w, roi_x2)
        roi_y2 = min(img_h, roi_y2)

        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return False, 0, "Error"

        roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Get Green channel mean
        g_mean = np.mean(roi[:, :, 1])
        self.signal_buffer.append(g_mean)
        
        if len(self.signal_buffer) > self.buffer_size:
            self.signal_buffer.pop(0)

        if len(self.signal_buffer) < self.buffer_size:
            progress = (len(self.signal_buffer) / self.buffer_size) * 100
            return False, 0, f"Init {progress:.0f}%" # Not enough data
        
        return self.analyze()

    def analyze(self):
        # Detrend
        data = np.array(self.signal_buffer)
        detrended = signal.detrend(data)
        
        # Filter (Bandpass 0.7Hz - 4Hz for 42-240 BPM)
        b, a = signal.butter(2, [0.7, 4.0], btype='bandpass', fs=self.fps)
        filtered = signal.filtfilt(b, a, detrended)
        
        # FFT
        fft = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(filtered), 1.0/self.fps)
        
        # Find peak
        idx = np.argmax(fft)
        peak_freq = freqs[idx]
        bpm = peak_freq * 60
        
        # Simple liveness check: if there is a strong peak in human HR range
        # This is a basic heuristic. Real rPPG is more complex.
        is_live = 45 < bpm < 200
        
        return is_live, bpm, "Done"
