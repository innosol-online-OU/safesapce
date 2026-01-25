
import os
import sys
import torch
import numpy as np
from PIL import Image
from core.protocols.latent_cloak import LatentCloak, ProtectionConfig

def test_defense():
    print("--- [DEBUG] Starting Defense Test ---", flush=True)
    
    # 1. Create Dummy Image (Red Square with a White "Face" Circle)
    # We need a "face" so the detector doesn't fail and skip protection.
    img = Image.new('RGB', (512, 512), color=(255, 0, 0))
    # Draw a face-like oval
    import cv2
    arr = np.array(img)
    cv2.ellipse(arr, (256, 256), (100, 150), 0, 0, 360, (255, 255, 255), -1)
    # Add "eyes" to make it detected as face (hopefully) by weak detectors, 
    # but InsightFace is strong. 
    # Actually, InsightFace will likely fail on this drawing.
    
    # Check if we should use a real image if available
    real_img_path = "/app/uploads/test_face.png"
    if os.path.exists(real_img_path):
        print(f"Using real image: {real_img_path}")
        img = Image.open(real_img_path).convert("RGB")
    else:
        print("Using synthetic image (Warning: Face detection might fail)", flush=True)
        img = Image.fromarray(arr)
        img.save("debug_input.png")

    # 2. Initialize
    print("Initializing LatentCloak...", flush=True)
    try:
        cloak = LatentCloak()
    except Exception as e:
        print(f"FATAL: Init failed: {e}", flush=True)
        return

    # 3. Protect
    print("Running protect()...", flush=True)
    config = ProtectionConfig(strength=100, face_boost=3.0)
    
    # We need a valid path for protect() because it opens the file
    img.save("debug_temp.png")
    
    try:
        out = cloak.protect("debug_temp.png", config)
    except Exception as e:
        print(f"FATAL: Protection crashed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    # 4. Verify
    arr_in = np.array(img).astype(float)
    arr_out = np.array(out).astype(float)
    mse = np.mean((arr_in - arr_out)**2)
    print(f"--- RESULT ---")
    print(f"MSE: {mse:.4f}")
    
    if mse < 1.0:
        print("FAIL: Image did not change significantly!", flush=True)
        
        # DEBUGGING INTERNAL STATE using direct access if possible
        # Check if detect_and_crop worked
        # We can't easily see local vars, but logs should have shown "Protect called"
    else:
        print("SUCCESS: Image modified.", flush=True)

if __name__ == "__main__":
    test_defense()
