
import os
import sys
from PIL import Image
import cv2

# Add . to path so we can import validator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.core.cloaking import CloakEngine
from validator import validate_image

def verify_dual_layer_defense():
    print("Initializing CloakEngine for Dual-Layer Verification...")
    engine = CloakEngine()
    
    # Create input
    input_path = "verify_input.png"
    # Create a dummy image with a "face-like" structure or just noise?
    # Ideally standard verification needs a face for PrivacyCloak to find one.
    # We will use the center-patch fallback of PrivacyCloak if no face found.
    # To test face detection, we'd need a real face image. 
    # For now, we rely on the PrivacyCloak fallback logic or expect warning from validator.
    
    img = Image.new('RGB', (512, 512), color = (100, 150, 200))
    # Draw simple "face" to maybe trick HAAR? Unlikely without real features.
    img.save(input_path)
    
    # 1. Test Privacy Shield + Compliance
    print("\n--- Testing: Privacy Shield + Compliance ---")
    out_privacy = "output_privacy.png"
    if os.path.exists(out_privacy): os.remove(out_privacy)
    
    success = engine.apply_defense(input_path, out_privacy, visual_mode="privacy", compliance=True)
    if success:
        print("✅ Generation Successful")
        validate_image(input_path, out_privacy, check_compliance=True, check_privacy=True)
    else:
        print("❌ Generation Failed")

    # 2. Test Art Shield + Compliance
    print("\n--- Testing: Art Shield + Compliance ---")
    out_art = "output_art.png"
    if os.path.exists(out_art): os.remove(out_art)
    
    success = engine.apply_defense(input_path, out_art, visual_mode="art", compliance=True)
    if success:
        print("✅ Generation Successful")
        validate_image(input_path, out_art, check_compliance=True, check_privacy=False)
    else:
        print("❌ Generation Failed")

if __name__ == "__main__":
    verify_dual_layer_defense()
