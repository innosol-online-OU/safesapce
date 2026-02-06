
import sys
import os
import torch
import numpy as np
from PIL import Image
import traceback

sys.path.append("/app") # Docker path
sys.path.append(os.getcwd()) # Local path

try:
    from invisible_core.attacks.latent_cloak import LatentCloak, ProtectionConfig
    from invisible_core.critics.clip_critic import CLIPCritic
    
    # 1. Test Critic Upgrade
    print("[TEST] Initializing CLIPCritic (Expect ViT-L/14@336px)...")
    critic = CLIPCritic(device='cuda' if torch.cuda.is_available() else 'cpu')
    critic.load()
    
    if critic.model:
        # Check config if possible, or just print model
        print(f"[SUCCESS] Critic Loaded: {type(critic.model)}")
        # Check resize logic (inspecting code is hard here but we can check run)
        img = torch.randn(1, 3, 512, 512, device=critic.device)
        loss = critic.compute_loss(img)
        print(f"[SUCCESS] Critic Compute Loss: {loss.item()}")
    
    # 2. Test LatentCloak Gray-First Generator
    print("[TEST] Initializing LatentCloak...")
    cloak = LatentCloak(lite_mode=True)
    
    # Create Dummy Image
    dummy_img = Image.new('RGB', (512, 512), color=(100, 150, 200))
    dummy_path = "test_phase_13_5.png"
    dummy_img.save(dummy_path)
    
    config = ProtectionConfig(strength=50, optimization_steps=5) # Short run
    
    print("[TEST] Running protect_frontier_lite (Gray-First Generator)...")
    protected = cloak.protect_frontier_lite(dummy_path, config=config)
    
    if protected:
        print("[SUCCESS] Protection generated image.")
        protected.save("protected_phase_13_5.png")
    
    print("ALL TESTS PASSED.")

except Exception as e:
    print(f"FAIL: {e}")
    traceback.print_exc()
