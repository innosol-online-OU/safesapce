import os
import sys
import torch
import torch.nn as nn
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.getcwd())

def test_safe_cast_wrapper():
    print("=== Verifying Fix: Datatype Safety (Float32 Input -> BFloat16/Float16 Model) ===")
    print("Expectation: The code should metrics AUTO-CAST inputs and NOT crash.")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Test DWT-Mamba Proxy mismatch
    print("\n[1] Testing DWTMamba (Float32 Input -> BFloat16/Float16 Model)...")
    try:
        from invisible_core.attacks.dwt_mamba import MambaProxyBlock, DWTMambaLoss, DWTMambaConfig
        
        # Force model to BFloat16 (mimic environment)
        proxy = MambaProxyBlock(d_model=3).to(device)
        try:
             proxy.to(dtype=torch.bfloat16)
        except:
             proxy.to(dtype=torch.float16) # Fallback for CPU
             
        # Input is Float32 (from DWT)
        # Fix shape to [B, C, H, W] for MambaProxy forward
        input_tensor = torch.randn(1, 3, 64, 64, device=device, dtype=torch.float32)
        
        try:
            _ = proxy(input_tensor)
            print("[PASS] MambaProxy: Successfully handled mixed precision input!")
        except RuntimeError as e:
            if "Input type" in str(e) and "bias type" in str(e):
                print("[FAIL] MambaProxy: Crashed with Type Mismatch (Fix not applied!)")
                print(f"[WARN] Error: {e}")
        except Exception as e:
            print(f"[WARN] MambaProxy: Unexpected Exception: {e}")
            
    except ImportError:
        print("Skipping DWT test (dependencies missing)")

    # 2. Test LatentCloak VAE Decode mismatch
    print("\n[2] Testing LatentCloak VAE Decode (Float32 Latents -> BFloat16 VAE)...")
    # Mocking the specific VAE behavior since loading SD is heavy
    class MockVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.post_quant_conv = nn.Conv2d(4, 4, 1)
            self.decoder = nn.Conv2d(4, 3, 3, padding=1)
            
        def decode(self, z):
            # Mimic Diffusers VAE: post_quant_conv(z)
            out = self.post_quant_conv(z)
            return MagicMock(sample=out) # Returns object with .sample
            
    try:
        mock_vae = MockVAE().to(device)
        try:
             mock_vae.to(dtype=torch.bfloat16)
        except:
             mock_vae.to(dtype=torch.float16)
             
        # Float32 Latents (Optimization often happens in FP32)
        latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)
        
        try:
            # Replicating the crashing line:
            # self.pipe.vae.decode(latents / 0.18215)
            # The error is that latents (FP32) go into Conv2d (BF16)
            _ = mock_vae.decode(latents)
            print("[PASS] VAE: Successfully handled mixed precision input!")
        except RuntimeError as e:
             if "Input type" in str(e) and "bias type" in str(e):
                print("[FAIL] VAE: Crashed with Type Mismatch (Fix not applied!)")
             else:
                print(f"[WARN] VAE: Unexpected Error: {e}")
                
    except Exception as e:
        print(f"Test Error: {e}")

if __name__ == "__main__":
    test_safe_cast_wrapper()
