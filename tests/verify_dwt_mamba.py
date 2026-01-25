
import torch
import sys
import os
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, os.getcwd())

from src.core.protocols.dwt_mamba import DWTMambaLoss, DWTMambaConfig

def test_dwt_mamba():
    print("Testing DWT-Mamba Integration...")
    
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU (might be slow but functional for check)")
        device = 'cpu'
    else:
        device = 'cuda'
        
    # 1. Init
    try:
        config = DWTMambaConfig()
        model = DWTMambaLoss(config, device=device).to(device)
        print("✅ DWTMambaLoss initialized.")
    except Exception as e:
        print(f"❌ Init failed: {e}")
        return

    if model.dwt is None:
        print("❌ DWT not available (pytorch_wavelets missing?).")
        return

    # 2. Create Dummy Tensors [B, 3, 512, 512]
    # 'adv_img' requires grad
    adv_img = torch.randn(1, 3, 512, 512, device=device, requires_grad=True)
    orig_img = torch.randn(1, 3, 512, 512, device=device, requires_grad=False)
    
    print("✅ Dummy tensors created.")

    # 3. Forward Pass
    try:
        loss, breakdown = model(adv_img, orig_img)
        print(f"✅ Forward pass success. Loss: {loss.item():.4f}")
        print(f"   Breakdown: {breakdown}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Backward Pass (Check Gradients)
    try:
        loss.backward()
        print("✅ Backward pass success.")
        
        if adv_img.grad is not None:
             grad_mag = adv_img.grad.abs().mean().item()
             print(f"✅ Gradients flowing! Mean magnitude: {grad_mag:.6f}")
             if grad_mag == 0:
                 print("⚠️ Warning: Gradients are zero.")
             else:
                 print("✅ Gradients are non-zero. Differentiability confirmed.")
        else:
             print("❌ No gradients found on input tensor.")
             
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dwt_mamba()
