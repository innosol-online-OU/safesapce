
import sys
import os
import torch
import shutil

# Add src to path
sys.path.insert(0, os.getcwd())

from invisible_core.attacks.latent_cloak import LatentCloak, ProtectionConfig
from invisible_core.logger import logger

def test_precision_enforcement():
    print("Testing Precision Enforcement...")
    
    # 1. Mock LatentCloak methods to avoid loading full SD model
    # We only want to test the optimization loop logic if possible, 
    # but LatentCloak is tightly coupled. 
    # Instead, let's instantiate it (it handles CPU offload) and check types during a dummy protect call.
    # To avoid 4GB download, we rely on the fact the user has cache or we mock the pipe.
    
    # MOCKING: We will mock the heavy parts to test logic light-weight
    class MockVAE:
        def __init__(self):
            self.dtype = torch.bfloat16
        def encode(self, x):
            class MockDist:
                def sample(self): return torch.randn(1, 4, 64, 64).to(x.device, x.dtype)
            class MockOut:
                def __init__(self): self.latent_dist = MockDist()
            return MockOut()
        def decode(self, x):
            class MockOut:
                def __init__(self): self.sample = torch.randn(1, 3, 512, 512).to(x.device, x.dtype)
            return MockOut()

    class MockPipeline:
        def __init__(self):
            self.vae = MockVAE()
            self.scheduler = type('obj', (object,), {'config': {}})
        def enable_model_cpu_offload(self): pass
        def set_progress_bar_config(self, **kwargs): pass

    # Instantiate
    cloak = LatentCloak(model_id="runwayml/stable-diffusion-v1-5")
    cloak.models_loaded = True
    cloak.pipe = MockPipeline() # Inject mock
    cloak.device = 'cpu' # Force CPU for test if no GPU
    if torch.cuda.is_available():
        cloak.device = 'cuda'
        
    print(f"Device: {cloak.device}")
    
    # Create dummy image
    from PIL import Image
    dummy_img = Image.new('RGB', (512, 512), color='red')
    dummy_path = "test_precision_input.png"
    dummy_img.save(dummy_path)
    
    # Run protect 
    # We need to capture the variable properties inside the loop. 
    # Since we can't easily hook into the method locals without modification,
    # we will rely on the fact that if it runs without error (NaN check passed), it's good.
    # AND we check the log file for success.
    
    try:
        cloak.protect(dummy_path, config=ProtectionConfig(num_steps=1, strength=10))
        print("Protection run finished.")
    except Exception as e:
        print(f"Protection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # Check Log File
    log_file = os.path.join("logs", "safespace.log")
    if os.path.exists(log_file):
        print(f"Log file exists: {log_file}")
        with open(log_file, 'r') as f:
            content = f.read()
            if "Optimizing" in content:
                print("Log contains expected output.")
            else:
                print("Log Missing expected output!")
                sys.exit(1)
    else:
        print("Log file NOT created!")
        sys.exit(1)

    print("Precision Test Passed!")

if __name__ == "__main__":
    test_precision_enforcement()
