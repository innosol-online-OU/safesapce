
import os
import sys
from PIL import Image
import torch

# Add . to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.core.cloaking import CloakEngine
from validator_stealth import StealthValidator

def verify_stealth_pipeline():
    print("Initializing CloakEngine for Stealth Verification...")
    engine = CloakEngine()
    
    # Create input
    input_path = "verify_stealth_input.png"
    # Gradient image for better texture testing
    img = Image.new('RGB', (512, 512), color=(0,0,0))
    for i in range(512):
        for j in range(512):
            img.putpixel((i, j), (i % 255, j % 255, (i+j)%255))
    img.save(input_path)
    
    print("\n--- Testing: Privacy Shield (Sparse Patch) + Compliance (Token Injection) ---")
    out_privacy = "output_stealth_privacy.png"
    if os.path.exists(out_privacy): os.remove(out_privacy)
    
    # Run Generation
    # Compliance=True triggers AdversarialTokenInjector
    success = engine.apply_defense(input_path, out_privacy, visual_mode="privacy", compliance=True)
    
    if success:
        print("✅ Generation Successful")
        val = StealthValidator()
        val.validate(input_path, out_privacy, check_compliance=True)
    else:
        print("❌ Generation Failed")

    print("\n--- Testing: Art Shield + Compliance (Token Injection) ---")
    out_art = "output_stealth_art.png"
    if os.path.exists(out_art): os.remove(out_art)
    
    success = engine.apply_defense(input_path, out_art, visual_mode="art", compliance=True)
    
    if success:
        print("✅ Generation Successful")
        # Reuse validator (models loaded)
        val.validate(input_path, out_art, check_compliance=True)
    else:
        print("❌ Generation Failed")

if __name__ == "__main__":
    verify_stealth_pipeline()
