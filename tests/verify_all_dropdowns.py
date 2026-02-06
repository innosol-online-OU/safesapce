import sys
import os
import numpy as np
from PIL import Image
import traceback

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add cwd to path for imports to work as if running from root
sys.path.insert(0, os.getcwd())

# Mock Streamlit session state
class MockSessionState(dict):
    def __getattr__(self, key):
        return self.get(key)
    def __setattr__(self, key, value):
        self[key] = value

class MockStreamlit:
    session_state = MockSessionState()
    def spinner(self, text):
        class SpinnerContext:
            def __enter__(self): pass
            def __exit__(self, *args): pass
        print(f"Streamlit Spinner: {text}")
        return SpinnerContext()
    
    def error(self, text):
        print(f"STREAMLIT ERROR: {text}")

sys.modules['streamlit'] = MockStreamlit()

from invisible_core.cloaking import CloakEngine

def create_dummy_image(path):
    print(f"Generating dummy image at {path}...")
    img = Image.new('RGB', (512, 512), color=(128, 128, 128))
    for i in range(100):
        x = np.random.randint(0, 512)
        y = np.random.randint(0, 512)
        img.putpixel((x, y), (255, 0, 0))
    img.save(path)
    return path

def verify_all_methods():
    print("--- Verifying Dropdown Methods ---")
    
    # 1. Setup
    input_path = "tests/verify_input_temp.png"
    create_dummy_image(input_path)
    output_base = "tests/verify_output"
    
    engine = CloakEngine()
    
    methods = [
        ("Frontier Lite", "frontier"),
        ("Liquid Warp", "liquid_17"),
        ("Resonant Ghost", "phantom_15"),
        ("General", "general")
    ]
    
    results = {}
    
    for name, profile in methods:
        print(f"\nTesting Method: {name} (Profile: {profile})")
        output_path = f"{output_base}_{profile}.png"
        
        # Configure Mock Session State for specific methods
        import streamlit as st
        st.session_state['phantom_strength'] = 50
        st.session_state['phantom_retries'] = 1
        st.session_state['phantom_background'] = 0.2
        if profile == "liquid_17":
            st.session_state['liquid_grid'] = 16
            st.session_state['liquid_tv'] = 0.01 # Using fix value just in case, or default 50
            st.session_state['liquid_steps'] = 10 # Short run for verification
            st.session_state['liquid_limit'] = 0.004
            st.session_state['liquid_blur'] = 15
        
        try:
            success, heatmap, metrics = engine.apply_defense(
                image_path=input_path,
                output_path=output_path,
                visual_mode="latent_diffusion",
                compliance=False, # Skip stego for speed if possible, or True to test it
                strength=50,
                target_profile=profile,
                max_retries=1
            )
            
            if success:
                print(f"✅ {name}: Success!")
                results[name] = "PASS"
            else:
                print(f"❌ {name}: Failed (Returned False)")
                results[name] = "FAIL"
                
        except Exception as e:
            print(f"❌ {name}: Exception - {e}")
            traceback.print_exc()
            results[name] = "CRASH"

    # Cleanup
    if os.path.exists(input_path):
        os.remove(input_path)
    # output files check?
    
    print("\n--- Summary ---")
    all_pass = True
    for name, res in results.items():
        print(f"{name}: {res}")
        if res != "PASS":
            all_pass = False
            
    if all_pass:
        print("\nAll dropdown methods operational.")
        exit(0)
    else:
        print("\nSome methods failed.")
        exit(1)

if __name__ == "__main__":
    verify_all_methods()
