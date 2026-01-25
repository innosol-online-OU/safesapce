#!/usr/bin/env python
"""
Phase 17.9c Verification Script: Granular Controls for Liquid Warp V2

Tests:
1. Checks that protect_liquid_warp_v2 accepts `asymmetry_strength` and `grid_size`.
2. Runs protection with different asymmetry levels (0.0 vs 0.3).
3. Runs protection with different grid sizes (14 vs 64).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image
import time
import torch

def create_test_image(path, size=(512, 512)):
    """Create a simple test image with a face-like pattern."""
    img = Image.new('RGB', size, color=(200, 180, 160))
    pixels = img.load()
    cx, cy = size[0] // 2, size[1] // 2
    # Eyes
    for dx in [-50, 50]:
        for dy in range(-10, 10):
            pixels[cx + dx, cy - 30 + dy] = (50, 40, 35)
    img.save(path)
    return path

def verify_controls():
    print("=" * 60)
    print("Phase 17.9c Verification: Granular Controls")
    print("=" * 60)
    
    test_input = "tests/verify_input_controls.png"
    create_test_image(test_input)
    
    # Import
    from invisible_core.attacks.latent_cloak import LatentCloak
    cloak = LatentCloak(lite_mode=True) # Avoid loading SD
    print("[OK] LatentCloak initialized")
    
    # Test 1: High Asymmetry (0.3)
    print("\n[TEST 1] High Asymmetry (0.3), Grid 16")
    try:
        res1 = cloak.protect_liquid_warp_v2(
            test_input,
            strength=75,
            grid_size=16,
            num_steps=20, 
            lr=0.01,
            asymmetry_strength=0.3
        )
        print("   [SUCCESS] High asymmetry run complete.")
    except Exception as e:
        print(f"   [FAIL] {e}")
        return False

    # Test 2: Zero Asymmetry (0.0), Grid 64
    print("\n[TEST 2] Zero Asymmetry (0.0), Grid 64")
    try:
        res2 = cloak.protect_liquid_warp_v2(
            test_input,
            strength=75,
            grid_size=64,
            num_steps=20,
            lr=0.01,
            asymmetry_strength=0.0
        )
        print("   [SUCCESS] Zero asymmetry/High Grid run complete.")
    except Exception as e:
        print(f"   [FAIL] {e}")
        return False

    # Verification
    # Ensure they ran and produced output
    if res1.size == (512, 512) and res2.size == (512, 512):
        print("\n[PASS] Granular controls verified successfully.")
        
        # Cleanup
        if os.path.exists(test_input): os.remove(test_input)
        return True
    
    return False

if __name__ == "__main__":
    success = verify_controls()
    exit(0 if success else 1)
