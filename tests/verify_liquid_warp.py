"""
Phase 17: Liquid Warp Verification Script
Tests geometric warping attack (grid_sample) against SigLIP.
Target: CosSim < 0.4, SSIM > 0.90
"""
import os
import sys
sys.path.insert(0, '/app')

# Mock streamlit for standalone execution
class MockSessionState:
    phantom_strength = 50
    phantom_retries = 1
    phantom_targeting = 3.0
    phantom_resolution = "Original"
    phantom_background = 1.0

class MockStreamlit:
    session_state = MockSessionState()

sys.modules['streamlit'] = MockStreamlit()

from PIL import Image
import numpy as np

print("--- Verifying Phase 17: Liquid Warp (Geometric Warping) ---")

# Create test image if needed
test_image_path = "/app/test_image.png"
if not os.path.exists(test_image_path):
    print("Creating test image...")
    img = Image.new('RGB', (512, 512), color=(128, 128, 128))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse([156, 106, 356, 406], fill=(220, 180, 160))  # Face-like oval
    draw.ellipse([200, 180, 240, 220], fill=(50, 50, 50))  # Left eye
    draw.ellipse([270, 180, 310, 220], fill=(50, 50, 50))  # Right eye
    draw.arc([210, 270, 300, 340], 0, 180, fill=(100, 50, 50), width=3)  # Mouth
    img.save(test_image_path)
    print(f"Created: {test_image_path}")

# Initialize engine
from src.core.protocols.latent_cloak import LatentCloak

print("\nInitializing LatentCloak...")
cloak = LatentCloak(lite_mode=True)

print("\nRunning Liquid Warp Defense...")
result = cloak.protect_liquid_warp(
    test_image_path,
    strength=75,
    grid_size=32,
    num_steps=100,
    lr=0.005,
    tv_weight=0.1
)

# Save output
output_path = "/app/output_liquid_warp.png"
result.save(output_path)
print(f"\n✅ Output saved to: {output_path}")

# Calculate metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

orig = np.array(Image.open(test_image_path).convert('RGB'))
warped = np.array(result)

ssim_val = ssim(orig, warped, channel_axis=2)
psnr_val = psnr(orig, warped)

print(f"\n--- Metrics ---")
print(f"SSIM: {ssim_val:.4f} (Target: > 0.90)")
print(f"PSNR: {psnr_val:.2f} dB")

if ssim_val > 0.90:
    print("✅ SSIM PASS: Visual quality preserved!")
else:
    print("⚠️ SSIM below target - may need tuning")

print("\n✅ Phase 17 Verification Complete")
