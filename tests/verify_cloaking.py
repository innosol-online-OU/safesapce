import os
import sys
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from invisible_core.cloaking import CloakEngine

def verify_cloaking():
    print("Initializing CloakEngine...")
    engine = CloakEngine()
    
    # Create a dummy image
    print("Creating test image...")
    input_path = "verify_input.png"
    output_path = "verify_output.png"
    
    img = Image.new('RGB', (800, 600), color = (100, 100, 255))
    img.save(input_path)
    
    print(f"Running cloaking on {input_path}...")
    try:
        success = engine.cloak_image(input_path, output_path)
        
        if success:
            print(f"✅ Cloaking successful! Output saved to {output_path}")
            
            # Verify the output exists and is a PNG
            if os.path.exists(output_path):
                out_img = Image.open(output_path)
                print(f"Output Format: {out_img.format}")
                print(f"Output Mode: {out_img.mode}")
                print("Verification passed.")
            else:
                print("❌ output file missing.")
        else:
            print("❌ Cloaking returned False.")
            
    except Exception as e:
        print(f"❌ Exception during verification: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    # if os.path.exists(input_path): os.remove(input_path)
    # if os.path.exists(output_path): os.remove(output_path)

if __name__ == "__main__":
    verify_cloaking()
