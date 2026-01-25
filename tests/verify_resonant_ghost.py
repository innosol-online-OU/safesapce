
import os
import sys
import numpy as np
from PIL import Image

sys.path.append("/app")

from src.core.cloaking import CloakEngine

def verify_resonant_ghost():
    print("--- Verifying Phase 16: Resonant Ghost (MI-FGSM + DIM) ---")
    
    # 1. Setup Input
    input_path = "target_test_image.png"
    if not os.path.exists(input_path):
        # Create dummy face
        img = Image.new('RGB', (512, 512), color=(100, 100, 100))
        img.save(input_path)
        
    output_path = "output_resonant_ghost.png"
    
    # 2. Setup Engine
    engine = CloakEngine()
    
    # 3. Configure Session State for Phantom
    # We use streamlit session state mocking if needed, or just defaults.
    # CloakEngine reads from st.session_state if available, else defaults.
    # Let's mock it to test parameters if possible, but CloakEngine.apply_defense uses explicit arguments for everything EXCEPT what it reads from st.session_state internally?
    # Wait, apply_defense READS st.session_state internally in lines 118-129.
    
    # Mock streamlit
    class MockStreamlit:
        session_state = {}
    sys.modules['streamlit'] = MockStreamlit()
    import streamlit as st
    
    st.session_state['phantom_strength'] = 50
    st.session_state['phantom_retries'] = 1 # 1 retry (loop)

    st.session_state['phantom_targeting'] = 3.0
    st.session_state['phantom_resolution'] = "512" # Resolution
    
    # 4. Run Protection
    print("Running Resonant Ghost Defense...")
    try:
        success, heatmap, metrics = engine.apply_defense(
            image_path=input_path,
            output_path=output_path,
            target_profile="phantom_15", # Triggers protect_phantom
            visual_mode="latent_diffusion"
        )
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Validate Results
    if success:
        print(f"✅ Success! Output saved to {output_path}")
        print("Metrics:", metrics)
        
        # Check generated heatmap
        if os.path.exists("uploads/debug_jnd_heatmap.png"):
             print("✅ JND Heatmap generated.")
        else:
             print("❌ JND Heatmap missing.")
             
        # Check Pixel Diff
        orig = Image.open(input_path).convert("RGB")
        out = Image.open(output_path).convert("RGB")
        diff = np.mean(np.abs(np.array(orig, dtype=float) - np.array(out, dtype=float)))
        print(f"Mean Pixel Perturbation: {diff:.4f}")
        
        if diff < 0.5:
             print("⚠️ Warning: Perturbation very low. Monitor logs for momentum checks.")
        else:
             print("✅ Perturbation active (Ghosting applied).")
             
    else:
        print("❌ Defense Failed.")

if __name__ == "__main__":
    verify_resonant_ghost()
