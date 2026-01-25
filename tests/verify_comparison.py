
import os
import sys
from PIL import Image
import torch
import cv2

# Add . to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.core.protocols.face_cloak import FaceCloak
from validator import Validator

def verify_comparison():
    print("--- Verifying Adv-Makeup Comparison Layer ---")
    
    # Check for weights
    weights_path = "models/adv_makeup/AdvMakeup_Gen.pth"
    if not os.path.exists(weights_path):
        print(f"⚠️ Weights not found at {weights_path}. Pipeline will use initialized (random) weights.")
    
    # Initialize
    try:
        cloak = FaceCloak(weights_path=weights_path)
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return

    # Create Input
    input_path = "verify_comparison_input.png"
    # Create a dummy image with a face-like structure using OpenCV to ensure detection
    # Drawing a simple face: Circle head, eyes, mouth
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[:] = (200, 200, 200) # Light gray bg
    cv2.circle(img, (256, 256), 100, (255, 220, 200), -1) # Skin tone circle
    cv2.circle(img, (220, 230), 10, (0, 0, 0), -1) # Left Eye
    cv2.circle(img, (290, 230), 10, (0, 0, 0), -1) # Right Eye
    cv2.ellipse(img, (256, 290), (40, 20), 0, 0, 180, (0, 0, 0), 5) # Mouth
    
    cv2.imwrite(input_path, img)
    
    # Run Protect
    output_path = "output_adv_makeup.png"
    try:
        pil_img = Image.open(input_path)
        protected_img = cloak.protect(pil_img)
        protected_img.save(output_path)
        print(f"✅ Protection applied. Saved to {output_path}")
    except Exception as e:
        print(f"❌ Protection Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run Validator
    # Note: Validator expects dlib face detection. My dummy face might strictly pass MediaPipe (FaceMesh) 
    # but maybe not dlib's HOG/CNN. 
    # Let's see if validator runs.
    print("\nRunning Validator...")
    val = Validator()
    val.validate(input_path, output_path)

import numpy as np
if __name__ == "__main__":
    verify_comparison()
