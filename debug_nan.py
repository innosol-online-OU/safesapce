
import torch
import numpy as np
from PIL import Image
from src.core.protocols.latent_cloak import LatentCloak, ProtectionConfig
import types

# 1. Instantiate Cloak
print("Initializing LatentCloak for Debugging...")
cloak = LatentCloak()

# 2. Define the Debug Version of protect (Monkeypatch)
def debug_protect_loop(self, image_path: str, config: ProtectionConfig = None, user_mask: np.ndarray = None):
    print(f"[DEBUG] Potect called on {image_path}...", flush=True)
    if not self.models_loaded: return Image.open(image_path)
    if config is None: config = ProtectionConfig()
        
    original_pil = Image.open(image_path).convert("RGB")
    crop_pil, bbox, face_mask_np = self.detect_and_crop(original_pil)
    if not crop_pil: return original_pil
    
    # Setup Latents
    orig_latents = self.get_latents(crop_pil)
    adv_latents = orig_latents.clone().detach()
    adv_latents.requires_grad = True
    
    optimizer = torch.optim.Adam([adv_latents], lr=0.02)
    
    steps = 5 # Just run a few steps to catch the explosion
    print(f"[DEBUG] Starting short optimization loop ({steps} steps)...")
    
    for i in range(steps):
        # --- DEBUG CHECK REQUESTED BY USER ---
        print(f"Step {i} | Latents Max: {adv_latents.max().item():.4f}, Min: {adv_latents.min().item():.4f}")
        
        optimizer.zero_grad()
        
        # Decode (mimicking the loop)
        decoded_img = self.pipe.vae.decode(adv_latents / 0.18215).sample
        
        # Simple Dummy Loss for provocation (maximize mean to explode gradients)
        loss = decoded_img.mean() * 1000.0 
        
        loss.backward()
        
        if adv_latents.grad is not None:
             grad_max = adv_latents.grad.max().item()
             grad_min = adv_latents.grad.min().item()
             print(f"   -> Grads Max: {grad_max:.4f}, Min: {grad_min:.4f}")
             
             if torch.isnan(adv_latents.grad).any() or torch.isinf(adv_latents.grad).any():
                 print("[DEBUG] ⚠️ NaN/Inf gradient detected!")
                 break
        
        optimizer.step()

    print("[DEBUG] Investigation Complete.")
    return original_pil

# 3. Apply Patch
cloak.protect = types.MethodType(debug_protect_loop, cloak)

# 4. Create Dummy Input
dummy_img = Image.new('RGB', (512, 512), color='red')
dummy_path = "debug_input.png"
dummy_img.save(dummy_path)

# 5. Run
cloak.protect(dummy_path)
