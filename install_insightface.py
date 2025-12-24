import os
import urllib.request
import subprocess
import sys

def install_insightface():
    # URL for a community-provided wheel for Python 3.12 on Windows
    # This is a common workaround. If this link dies, we must fall back to build tools.
    # Source: https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp312-cp312-win_amd64.whl
    wheel_url = "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp312-cp312-win_amd64.whl"
    wheel_name = "insightface-0.7.3-cp312-cp312-win_amd64.whl"
    
    print(f"Downloading {wheel_name}...")
    try:
        urllib.request.urlretrieve(wheel_url, wheel_name)
        print("Download complete.")
        
        print("Installing wheel...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_name])
        print("Installation successful.")
        
        # Cleanup
        os.remove(wheel_name)
    except Exception as e:
        print(f"Failed to install insightface from wheel: {e}")
        print("Please install 'Desktop development with C++' via Visual Studio Build Tools.")

if __name__ == "__main__":
    install_insightface()
