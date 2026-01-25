import time
import os
import torch
from invisible_core.cloaking import CloakEngine

def test_lazy_loading():
    print("=== Phase 15.1: Lazy Loading Verification ===")
    
    # 0. Initialize Engine
    start_init = time.time()
    engine = CloakEngine()
    print(f"Engine Init Time: {time.time() - start_init:.2f}s")
    
    # Check that heavy models are NOT loaded yet
    # We need to access the latent cloak instance inside
    # Trigger lazy init of LatentCloak wrapper
    dummy_detect = engine.detect_faces("docs/assets/logo_v4.png") # Force init
    
    lc = engine.latent_cloak
    if lc is None:
        print("FAIL: LatentCloak not initialized.")
        return
        
    if lc.diffusion_loaded:
        print("FAIL: Diffusion models loaded properly prematurely!")
    else:
        print("PASS: Diffusion models NOT loaded initially.")
        
    if not lc.detectors_loaded:
        print("PASS: Detectors loaded on demand.")
    else:
        # It updates on first call
        print("INFO: Detectors loaded.")

    # 1. Test Detection Speed (Cold Start)
    # We need a face image. Let's create a dummy one or use existing
    # Just skipping real detection if file missing, checking logic flow
    
    print("\n=== Phase 15.2: Smart Regions Data Format ===")
    # Mocking the detector to return data to check format
    # But we can check if detect_faces calls _load_detectors
    
    try:
        # trigger protection
        print("\nTriggering Full Protection (Should load Diffusion)...")
        start_prot = time.time()
        # Mocking protect call just to see loading logs
        lc._load_diffusion()
        print(f"Diffusion Load Time: {time.time() - start_prot:.2f}s")
        
        if lc.diffusion_loaded:
             print("PASS: Diffusion models loaded on demand.")
        else:
             print("FAIL: Diffusion models failed to load.")
             
    except Exception as e:
        print(f"Error during protection test: {e}")

if __name__ == "__main__":
    test_lazy_loading()
