
import os
import torch
import numpy as np
from PIL import Image
from src.core.protocols.latent_cloak import LatentCloak, ProtectionConfig
from src.core.logger import logger

def verify_dwt_attack():
    print("--- Verifying Phase 5.2 DWT-Mamba Attack ---")
    
    # Create dummy image (Person-like? Or just noise?)
    # Attacker needs something to detect.
    # Let's create a dummy image that might trigger "person" or just standard noise.
    # A solid color might not trigger YOLO.
    # Let's try to load a real asset if possible, or generate random noise that mimics nature?
    # Simply using a random noise image.
    img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    img_path = "dummy_input.png"
    img.save(img_path)
    
    # Initialize Cloak
    # Lite mode = True (Frontier Lite)
    cloak = LatentCloak(lite_mode=True)
    
    # Config for Frontier Lite (Phase 4 Lite)
    # Note: protect_frontier_lite overrides config internally for parameters,
    # but we pass one anyway to match signature.
    config = ProtectionConfig(target_profile='frontier', defense_mode='aggressive')
    
    # Create a dummy image or load one
    img_path = "debug_input.png"
    if not os.path.exists(img_path):
        print("debug_input.png not found, creating noise...")
        # Create a dummy image for testing
        img = Image.new('RGB', (512, 512), color = (73, 109, 137))
        img.save(img_path)
    
    # Run protection
    print("[TEST] Running protect_frontier_lite...")
    
    # Run the function
    try:
        protected_img = cloak.protect_frontier_lite(img_path, config=config)
        print("[TEST] Protection complete.")
    except Exception as e:
        print(f"[TEST] Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_dwt_attack()
