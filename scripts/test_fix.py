import sys
import os
sys.path.append(os.getcwd())
import torch

from src.core.protocols.selective_ensemble import CLIPViTBaseSurrogate

def test_dtype_mismatch():
    print("Testing BFloat16 support in SelectiveEnsemble (CLIP)...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Initialize Surrogate
    surrogate = CLIPViTBaseSurrogate(device=device)
    surrogate.load()
    
    # 2. Force model to BFloat16 (mimic the error condition)
    print(f"Converting model to BFloat16 on {device}...")
    surrogate.model.to(dtype=torch.bfloat16)
    
    # 3. Create input tensor (Float32) - MIMIC THE INPUT TYPE FROM THE ERROR
    input_tensor = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float32)
    
    # 4. Create dummy features
    # CLIP features size is 512 for base
    orig_features = torch.randn(1, 512, device=device, dtype=torch.bfloat16) # Match model output type
    
    # 5. Run compute_loss
    try:
        loss = surrogate.compute_loss(input_tensor, orig_features)
        print(f"Success! Loss computed: {loss.item()}")
        print(f"Loss dtype: {loss.dtype}")
        
        # Verify it handled the mix
        if loss.dtype == torch.bfloat16 and torch.cuda.is_available():
             print("Verified: Operation performed in BFloat16")
        elif loss.dtype == torch.float32:
             print("Verified: Output is Float32 (loss usually is)")
             
    except RuntimeError as e:
        print(f"FAILED with RuntimeError: {e}")
        if "Input type" in str(e) and "bias type" in str(e):
             print("CONFIRMED: This represents the error.")
        else:
             print("Different error.")
    except Exception as e:
        print(f"FAILED with Exception: {e}")

if __name__ == "__main__":
    test_dtype_mismatch()
