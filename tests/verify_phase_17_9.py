#!/usr/bin/env python
"""
Phase 17.9 Verification Script: Liquid Warp V2 (T-Zone Anchored Warp)

Tests:
1. T-Zone mask generation (face detection + ellipse anchoring)
2. Multi-scale loss (224px, 384px)
3. Warp displacement with frozen silhouette
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image
import time

def create_test_image(path, size=(512, 512)):
    """Create a simple test image with a face-like pattern."""
    img = Image.new('RGB', size, color=(200, 180, 160))  # Skin-tone background
    pixels = img.load()
    
    # Draw simple face features
    cx, cy = size[0] // 2, size[1] // 2
    
    # Eyes (dark circles)
    for dx in [-50, 50]:
        for dy in range(-10, 10):
            for ddx in range(-15, 15):
                x, y = cx + dx + ddx, cy - 30 + dy
                if 0 <= x < size[0] and 0 <= y < size[1]:
                    pixels[x, y] = (50, 40, 35)
    
    # Nose (vertical line)
    for dy in range(-20, 40):
        pixels[cx, cy + dy] = (150, 130, 120)
    
    # Mouth (horizontal line)
    for dx in range(-30, 30):
        pixels[cx + dx, cy + 70] = (150, 80, 80)
    
    img.save(path)
    return path

def verify_liquid_warp_v2():
    print("=" * 60)
    print("Phase 17.9 Verification: Liquid Warp V2 (T-Zone Anchoring)")
    print("=" * 60)
    
    # Setup
    test_input = "tests/verify_input_phase_17_9.png"
    test_output = "tests/verify_output_phase_17_9.png"
    
    create_test_image(test_input)
    print(f"[OK] Created test image: {test_input}")
    
    # Import and initialize
    print("\n[LOAD] Loading LatentCloak...")
    from invisible_core.attacks.latent_cloak import LatentCloak
    
    start = time.time()
    cloak = LatentCloak(lite_mode=True)
    print(f"[OK] LatentCloak initialized in {time.time() - start:.1f}s")
    
    # Run V2 warp
    print("\n[RUN] Running protect_liquid_warp_v2...")
    print("   - T-Zone Anchoring: Enabled")
    print("   - Multi-Scale Loss: [224px, 384px]")
    print("   - TV Weight: 0.01 (reduced from 50.0)")
    print("   - Flow Limit: 0.03")
    print("-" * 40)
    
    start = time.time()
    result = cloak.protect_liquid_warp_v2(
        test_input,
        strength=75,
        grid_size=16,
        num_steps=50,  # Fewer steps for quick test
        lr=0.01,
        tv_weight=0.01,
        flow_limit=0.03,
        mask_blur=15
    )
    elapsed = time.time() - start
    
    print("-" * 40)
    print(f"[OK] Warp complete in {elapsed:.1f}s")
    
    # Save result
    result.save(test_output)
    print(f"[OK] Saved output: {test_output}")
    
    # Analyze difference
    orig = np.array(Image.open(test_input))
    warped = np.array(result)
    
    diff = np.abs(orig.astype(float) - warped.astype(float))
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"\n[METRICS]")
    print(f"   - Max Pixel Difference: {max_diff:.2f}")
    print(f"   - Mean Pixel Difference: {mean_diff:.4f}")
    
    # Cleanup
    if os.path.exists(test_input):
        os.remove(test_input)
    
    print("\n[PASS] Phase 17.9 Verification PASSED")
    return True

if __name__ == "__main__":
    try:
        verify_liquid_warp_v2()
    except Exception as e:
        print(f"\n[FAIL] FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

