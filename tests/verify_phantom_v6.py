import torch
import os
import sys
import numpy as np
from PIL import Image

# Ensure we can import from src
sys.path.append("/app")

from invisible_core.cloaking import CloakEngine

def test_phantom_v6():
    engine = CloakEngine()
    
    # Create or load a sizeable test image
    img_path = "/app/uploads/test_v6.png"
    # Create a simple image with a face-like structure (or just any image)
    Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)).save(img_path)
    
    # 1. Test Auto-Detection
    print("\n--- Testing Auto-Detection ---")
    boxes = engine.detect_faces(img_path)
    print(f"Detected {len(boxes)} faces (random noise might not yield results, but testing call safety).")
    
    # 2. Test Tiered Intensity
    print("\n--- Testing Tiered Intensity ---")
    # Mock user mask (Precision Targeting)
    user_mask = np.zeros((512, 512), dtype=np.float32)
    user_mask[100:200, 100:200] = 1.0 # Mock target zone
    
    output_path = "/app/uploads/test_v6_protected.png"
    
    # Mock session state
    import streamlit as st
    st.session_state['phantom_strength'] = 50
    st.session_state['phantom_retries'] = 1
    st.session_state['phantom_targeting'] = 5.0 # High Target
    st.session_state['phantom_background'] = 0.1 # Very low background
    st.session_state['phantom_resolution'] = "Original"
    
    success, msg, stats = engine.apply_defense(
        img_path, 
        output_path, 
        visual_mode="latent_diffusion",
        target_profile="PHANTOM",
        user_mask=user_mask,
        background_intensity=0.1 # Passed explicitly or extracted
    )
    
    print(f"Result: {success}, {msg}")
    
    # Verify heatmap logic (internal log check would be ideal, but here we check existence)
    heatmap_path = "/app/uploads/debug_jnd_heatmap.png"
    if os.path.exists(heatmap_path):
        print(f"Success: Tiered Heatmap generated at {heatmap_path}")
    else:
        print("Failure: Heatmap NOT generated.")

if __name__ == "__main__":
    test_phantom_v6()
