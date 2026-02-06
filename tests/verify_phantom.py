
import torch
import numpy as np
from PIL import Image
try:
    from invisible_core.attacks.latent_cloak import LatentCloak, _get_siglip_critic
except ImportError as e:
    print(f"Import Error: {e}")
    exit(1)

def test_phantom():
    print("--- Verifying Phase 15 (Phantom Pixel) ---")
    
    # 1. Test SigLIP Loading
    try:
        import timm
        print("Timm found.")
    except ImportError:
        print("❌ TIMM NOT FOUND. Phase 15 will fail.")
        return

    # Initialize LatentCloak in Lite Mode
    cloak = LatentCloak(lite_mode=True)
    
    # 2. Compute JND Mask
    print("Testing JND Mask Computation...")
    # Create dummy image (hair texture vs smooth)
    h, w = 512, 512
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Add some noise/edges
    img[100:200, 100:200] = np.random.randint(0, 255, (100, 100, 3))
    
    img_pil = Image.fromarray(img)
    img_tensor = torch.tensor(img).float().permute(2,0,1).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu') / 255.0
    
    mask = cloak.compute_jnd_mask(img_tensor)
    print(f"JND Mask Mean: {mask.mean().item():.4f} (Should be > 0.005)")
    
    if mask.mean() < 0.005:
        print("[FAIL] JND Mask too low.")
    else:
        print("[PASS] JND Mask Test Passed.")

    # 3. Test Protect Loop
    print("Testing Protect Loop (Dry Run)...")
    try:
        # Create a real temporary file
        img_pil.save("test_input_phantom.png")
        
        # We need to manually trigger the lazy load here to check if it crashes
        # But protect_phantom calls it.
        
        # We can't easily run full protect without model download (timm will download).
        # Assuming internet access or cache.
        
        # Let's try to load the model explicitly first
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
             _get_siglip_critic(device)
             print("[PASS] SigLIP Loaded Successfully.")
        except Exception as e:
             print(f"[FAIL] SigLIP Load Failed: {e}")
             return

        # Run protection
        out = cloak.protect_phantom("test_input_phantom.png")
        out.save("test_output_phantom.png")
        print("[PASS] Protect Phantom Run Completed.")
        
    except Exception as e:
        print(f"[FAIL] Protect Loop Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_phantom()
