
import os
import torch
from src.core.protocols.latent_cloak import LatentCloak, ProtectionConfig
from PIL import Image
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LatentCloak")

def test_bandit_ensemble():
    print("Testing Phase 11: Bandit Ensemble...")
    
    # Initialize Engine (Lite Mode to skip SD load if possible, but Frontier Lite needs CLIP/YOLO)
    # The protect_frontier_lite uses YOLO and CLIP.
    # We can try initializing with lite_mode=True, but check if those specific lazy loaders work.
    # LatentCloak init: if lite_mode=True, skips SD.
    # protect_frontier_lite doesn't use self.pipe (SD)?
    # Let's check the code I modified.
    # It converts to tensor: img_tensor = torch.tensor...
    # It calls _get_clip_critic()
    # It calls _get_dwt_mamba()
    # It does NOT seem to use self.pipe in the snippet I saw (Phase 11 update).
    # Wait, Phase 10 loop (lines 923+) used `adv_img` and `img_tensor`.
    # It seems safe to use lite_mode=True if we don't need SD VAE.
    # Wait, does it use VAE? 
    # "img_tensor = torch.tensor(img_np)..." - This is pixel space optimization?
    # Yes, "Optimizes in PIXEL SPACE to break segmentation masks." (line 1013 of legacy, but 780 of current says "Perceptual Adaptive Masking").
    # The loop optimizes `perturbation` added to `img_tensor`. So it is Pixel Space.
    # So VAE is not needed. lite_mode=True is fine.
    
    engine = LatentCloak(lite_mode=True)
    
    # Load/Create a dummy image
    if os.path.exists("test_image.png"):
        img_path = "test_image.png"
    else:
        # Create a dummy image
        img = Image.new('RGB', (512, 512), color = (73, 109, 137))
        img.save("test_image.png")
        img_path = "test_image.png"
        
    print(f"Target: {img_path}")
    
    # Config
    config = ProtectionConfig(
        target_profile='frontier',
        defense_mode='aggressive'
    )
    
    # Run Protection
    try:
        # We need to call protect_frontier_lite
        # Note: protect_frontier_lite uses YOLO ("yolov8n-seg.pt").
        # Ensure model exists or is downloaded.
        
        protected_img = engine.protect_frontier_lite(img_path, config=config)
        
        output_path = "verify_bandit_output.png"
        protected_img.save(output_path)
        print(f"Success! Output saved to {output_path}")
        
    except Exception as e:
        print(f"Error during protection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bandit_ensemble()
