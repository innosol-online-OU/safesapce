import torch
import os
import sys
import numpy as np
from PIL import Image

# Ensure we can import from src
sys.path.append("/app")

from src.core.cloaking import CloakEngine

def test_phantom_v5():
    engine = CloakEngine()
    
    # Create or load a sizeable test image
    img_path = "/app/uploads/test_v5.png"
    Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)).save(img_path)
    
    # Create a user mask (Precision Targeting)
    # A white square in the center (100x100)
    user_mask = np.zeros((512, 512), dtype=np.float32)
    user_mask[200:300, 200:300] = 1.0
    
    output_path = "/app/uploads/test_v5_protected.png"
    
    # Mock session state for testing
    import streamlit as st
    
    print("\n--- Testing Phase 15.Y: Precision & Scale ---")
    st.session_state['phantom_strength'] = 50
    st.session_state['phantom_retries'] = 1
    st.session_state['phantom_targeting'] = 3.0 # 3x boost in mask
    st.session_state['phantom_resolution'] = 224 # Optimize at 224px
    
    # We call apply_defense which extracts from session state
    success, msg, stats = engine.apply_defense(
        img_path, 
        output_path, 
        visual_mode="latent_diffusion",
        target_profile="PHANTOM",
        user_mask=user_mask
    )
    
    print(f"Result: {success}, {msg}")
    
    # Check if heatmap exists
    heatmap_path = "/app/uploads/debug_jnd_heatmap.png"
    if os.path.exists(heatmap_path):
        print(f"Success: Heatmap generated at {heatmap_path}")
        # Note: In a real test, one would inspect the pixel values of the heatmap
        # to ensure the center square is brighter (boosted).
    else:
        print("Failure: Heatmap NOT generated.")

if __name__ == "__main__":
    test_phantom_v5()
