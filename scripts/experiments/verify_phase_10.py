import torch
import numpy as np
from PIL import Image
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("."))

from invisible_core.attacks.latent_cloak import LatentCloak, ProtectionConfig
from invisible_core.util.ssim_loss import SSIMLoss
from invisible_core.logger import logger

def test_phase_10():
    print("--- Verifying Phase 10 Implementation ---")
    
    # 1. Setup
    LC = LatentCloak(lite_mode=True)
    
    # Create a dummy image (Gradient + Noise) to detect edges
    W, H = 512, 512
    img_np = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            img_np[y, x] = [x % 255, y % 255, (x+y) % 255]
            
    # Add a "Hair" (High Freq) line
    img_np[100:110, :, :] = 255
    # Add a "Skin" (Smooth) area
    img_np[200:400, 200:400, :] = 128
    
    dummy_path = "test_phase_10_input.png"
    Image.fromarray(img_np).save(dummy_path)
    
    # 2. Test Mask Generation
    print("\n[Test 1] Generating Perceptual Mask...")
    img_pil = Image.open(dummy_path).convert("RGB")
    mask = LC.aggregate_high_freq(img_pil)
    
    mask_np = mask[0,0].cpu().numpy()
    mean_val = mask_np.mean()
    max_val = mask_np.max()
    print(f"Mask Stats: Mean={mean_val:.4f}, Max={max_val:.4f}")
    
    if max_val > 0.4:
        print("[PASS] High frequency features detected (Mask Max > 0.4)")
    else:
        print(f"[FAIL] Mask values too low (Max={max_val:.4f})")
        
    # Check "Skin" area (should be low)
    skin_val = mask_np[300, 300]
    print(f"Smooth Area Value: {skin_val:.4f}")
    if skin_val < 0.2:
        print("[PASS] Smooth area has low mask value")
    else:
        print(f"[FAIL] Smooth area value too high ({skin_val:.4f})")

    # 3. Test Full Protection Loop
    print("\n[Test 2] Running Phase 10 Protection Loop...")
    config = ProtectionConfig(strength=100) # Max strength to test limits
    
    output_path = "test_phase_10_output.png"
    protected_img = LC.protect_frontier_lite(dummy_path, config=config)
    protected_img.save(output_path)
    print(f"Saved output to {output_path}")
    
    # 4. Verify SSIM
    print("\n[Test 3] Verifying SSIM Consistency...")
    ssim_fn = SSIMLoss()
    
    t_orig = torch.tensor(np.array(img_pil).astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
    t_prot = torch.tensor(np.array(protected_img).astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
    
    loss = ssim_fn(t_prot, t_orig)
    ssim = 1.0 - loss.item()
    print(f"Final SSIM: {ssim:.4f}")
    
    if ssim > 0.90:
        print("[PASS] Visual quality preserved (SSIM > 0.90)")
    else:
        print(f"[FAIL] Visual quality degraded (SSIM={ssim:.4f})")
        
    print("\n--- Phase 10 Verification Complete ---")

if __name__ == "__main__":
    try:
        test_phase_10()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
