import time
import torch
import numpy as np
from PIL import Image
import os
import sys

# Ensure src is in path
sys.path.append(os.getcwd())

from invisible_core.attacks.latent_cloak import LatentCloak

def verify_pure_phantom():
    print("--- Verifying Phase 16: Pure Phantom ---")
    
    # 1. Initialize
    print("[1] Initializing LatentCloak...")
    start_init = time.time()
    cloak = LatentCloak(device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Initialized in {time.time() - start_init:.2f}s")
    
    # 2. Create Dummy Image
    print("[2] Creating Dummy Image (1024x1024)...")
    img = Image.new('RGB', (1024, 1024), color=(100, 150, 200))
    # Add simple pattern to detect
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse((400, 400, 600, 700), fill=(200, 150, 150)) # Fake Face
    img_path = "temp_verify_phantom.png"
    img.save(img_path)
    
    # 3. Run Protect Phantom (Pure Pixel)
    print("[3] Running protect_phantom (First Run - heavy load)...")
    start_run = time.time()
    try:
        # Load resources happens inside
        res = cloak.protect_phantom(
            img_path, 
            strength=50, 
            retries=3, 
            targeting_intensity=3.0,
            resolution="384",
            background_intensity=0.2
        )
        end_run = time.time()
        duration = end_run - start_run
        print(f"    Execution Time: {duration:.2f}s")
        
        if duration > 30:
            print("    ⚠️ WARNING: Execution > 30s. Check optimization loop or loading overhead.")
        else:
            print("    ✅ Speed Check Passed (<30s).")
            
        # 4. Check Output
        res.save("temp_verify_phantom_out.png")
        print("    Output saved to temp_verify_phantom_out.png")
        
        # Simple diff check
        diff = np.mean(np.abs(np.array(img) - np.array(res)))
        print(f"    Mean Pixel Diff: {diff:.2f}")
        if diff > 1.0:
            print("    ✅ Image Modified (Noise applied).")
        else:
            print("    ❌ Image Unchanged (Optimization failed).")
            
        # 5. Check dependencies (Ensure SD is GONE)
        if hasattr(cloak, 'pipe') and cloak.pipe is not None:
             print("    ❌ ERROR: Stable Diffusion Pipe found! Phase 16 failed.")
        else:
             print("    ✅ Clean Architecture: No Stable Diffusion pipe found.")

    except Exception as e:
        print(f"    ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_pure_phantom()
