
import os
import sys
import numpy as np
import torch
from PIL import Image

# Add src to path
sys.path.append("/app")
# Also add current directory just in case
sys.path.append(os.getcwd())

from src.core.cloaking import CloakEngine

def verify_phantom_light():
    print("--- Verifying Phase 14: Phantom Light Generator (Inside Frontier Profile) ---")
    
    # 1. Setup Engine
    engine = CloakEngine()
    
    # 2. Setup Input Image
    # Try to use target_test_image.png if exists, else create a random one
    input_path = "target_test_image.png"
    if not os.path.exists(input_path):
        print(f"Warning: {input_path} not found. Creating dummy face image.")
        input_path = "temp_test_face.png"
        # Create a 512x512 image with a 'face' (yellow circle) to trigger detectors
        img = Image.new('RGB', (512, 512), color=(50, 50, 50))
        import cv2
        img_np = np.array(img)
        # Draw a face-ish shape
        cv2.circle(img_np, (256, 256), 100, (200, 200, 255), -1) # Face
        cv2.circle(img_np, (220, 230), 10, (0, 0, 0), -1) # Eye L
        cv2.circle(img_np, (290, 230), 10, (0, 0, 0), -1) # Eye R
        Image.fromarray(img_np).save(input_path)
    
    output_path = "output_phantom_light_verify.png"
    
    # 3. Apply Defense with Frontier Profile
    # This should trigger protect_frontier_lite -> Phase 14 Phantom Light
    print(f"Applying defense on {input_path}...")
    
    success, heatmap, metrics = engine.apply_defense(
        image_path=input_path,
        output_path=output_path,
        target_profile="frontier", # Triggers Phase 14
        strength=50, # Should affect eps_map constraints
        visual_mode="latent_diffusion"
    )
    
    # 4. Analyze Results
    print(f"\nDefense Success: {success}")
    if success:
        print(f"Output saved to: {output_path}")
        print("Metrics:", metrics)
        
        # Check if output actually differs (Simple check)
        orig = Image.open(input_path).convert("RGB")
        out = Image.open(output_path).convert("RGB")
        diff = np.mean(np.abs(np.array(orig, dtype=float) - np.array(out, dtype=float)))
        print(f"Mean Pixel Difference: {diff:.4f}")
        
        if diff < 0.1:
            print("FAILURE: Output is identical to input! Generator did not apply noise.")
        else:
            print("SUCCESS: Noise applied.")

if __name__ == "__main__":
    verify_phantom_light()
