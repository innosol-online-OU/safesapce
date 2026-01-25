import torch
import os
import sys

# Ensure we can import from src
sys.path.append("/app")

from src.core.cloaking import CloakEngine
from src.core.protocols.latent_cloak import ProtectionConfig

def test_phantom_manual():
    engine = CloakEngine()
    
    # Mock some image
    dummy_img_path = "/app/uploads/test_phantom.png"
    if not os.path.exists(dummy_img_path):
        import numpy as np
        from PIL import Image
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).save(dummy_img_path)
    
    output_path = "/app/uploads/test_phantom_protected.png"
    
    # We need to mock session state since apply_defense uses it
    import streamlit as st
    class MockSessionState(dict):
        def __getattr__(self, name): return self.get(name)
        def __setattr__(self, name, value): self[name] = value
    
    # Streamlit session state is a singleton or managed by streamlit
    # For testing, we can just patch it if we are running in a script
    # But CloakEngine imports streamlit inside apply_defense
    
    print("\n--- Testing Strength 10 (Stealth) ---")
    # Set session state mock
    st.session_state['phantom_strength'] = 10
    st.session_state['phantom_retries'] = 2
    
    # Note: Optimization steps are 40 in production, maybe we reduce for quick test?
    # Actually, let's just run it.
    
    success, msg, stats = engine.apply_defense(
        dummy_img_path, 
        output_path, 
        visual_mode="latent_diffusion", # This triggers the loop
        strength=50, # Global strength (ignored for Phantom)
        target_profile="PHANTOM" # This selects Phantom
    )
    
    print(f"Result: {success}, {msg}")
    
    print("\n--- Testing Strength 100 (Power) ---")
    st.session_state['phantom_strength'] = 100
    st.session_state['phantom_retries'] = 1
    
    success, msg, stats = engine.apply_defense(
        dummy_img_path, 
        output_path, 
        visual_mode="latent_diffusion",
        target_profile="PHANTOM"
    )
    
    print(f"Result: {success}, {msg}")

if __name__ == "__main__":
    test_phantom_manual()
