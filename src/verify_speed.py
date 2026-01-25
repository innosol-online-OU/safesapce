
import sys
import time
from unittest.mock import MagicMock
import os

# --- MOCK HEAVY DEPENDENCIES BEFORE IMPORT ---
# This prevents loading torch, diffusers, etc. which take seconds/minutes
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()
sys.modules["diffusers"] = MagicMock()
sys.modules["insightface"] = MagicMock()
sys.modules["insightface.app"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["lpips"] = MagicMock()
sys.modules["skimage"] = MagicMock()
sys.modules["skimage.metrics"] = MagicMock()

# Mock Protocols to avoid them importing real dependencies
dummy_lc_module = MagicMock()
class DummyLatentCloak:
    def __init__(self):
        print("DummyLatentCloak Initialized!")
    def protect(self, *args, **kwargs):
        # Return a dummy object that has a .save() method to mock Image
        img = MagicMock()
        img.save = MagicMock()
        img.size = (512, 512) # Fix for resize error
        return img
    def protect_frontier(self, *args, **kwargs):
        img = MagicMock()
        img.save = MagicMock()
        img.size = (512, 512) # Fix for resize error
        return img
    def add_trust_badge(self, img, **kwargs):
        return img
    def compute_similarity(self, *args):
        return 0.1

dummy_lc_module.LatentCloak = DummyLatentCloak
dummy_lc_module.ProtectionConfig = MagicMock()
sys.modules["src.core.protocols.latent_cloak"] = dummy_lc_module

dummy_qc_module = MagicMock()
class DummyQwen:
    def __init__(self): 
        self.use_api = False
    def critique(self, path):
        return True, "Passed", {}
dummy_qc_module.QwenCritic = DummyQwen
sys.modules["src.core.critics.qwen_critic"] = dummy_qc_module

dummy_stego_module = MagicMock()
sys.modules["src.core.protocols.stego_text"] = dummy_stego_module

# Add current directory to path
sys.path.insert(0, os.getcwd())

# Now safe to import engine (it will use mocks)
from src.core.cloaking import CloakEngine

def test_speed_optimization():
    print("Testing Speed Optimization (Model Persistence)... [FAST MOCK MODE]")
    
    # Initialize Engine
    engine = CloakEngine()
    
    # 1. Verify latent_cloak is None initially
    if engine.latent_cloak is not None:
        print("ERROR: latent_cloak should be None initially.")
        sys.exit(1)
        
    print("--- Run 1 (Should Init) ---")
    start = time.time()
    # Call apply_defense
    # path doesn't need to exist because we mocked PIL/Image.open in the engine?
    # Wait, engine imports PIL.Image.open. We didn't mock PIL.
    # So we need real files or mock PIL too.
    # Let's use real dummy files to be safe with PIL, it's fast.
    
    input_path = "test_speed_input.png"
    output_path = "test_speed_output.png"
    
    # Ensure dummy input exists
    if not os.path.exists(input_path):
        from PIL import Image
        Image.new('RGB', (64, 64)).save(input_path)
        
    engine.apply_defense(input_path, output_path, strength=10, compliance=False)
    dur1 = time.time() - start
    print(f"Run 1 Duration: {dur1:.4f}s")
    
    # 2. Check Logic: Did it persist?
    if engine.latent_cloak is None:
        print("❌ FAILURE: latent_cloak was unloaded!")
        sys.exit(1)
    else:
        print("✅ SUCCESS: latent_cloak persisted.")
        
    print("--- Run 2 (Should use Cache) ---")
    start = time.time()
    engine.apply_defense(input_path, output_path, strength=10, compliance=False)
    dur2 = time.time() - start
    print(f"Run 2 Duration: {dur2:.4f}s")
    
    # Cleanup
    if os.path.exists(input_path): os.remove(input_path)
    if os.path.exists(output_path): os.remove(output_path)
    
    print("Speed Optimization Verified!")

if __name__ == "__main__":
    test_speed_optimization()
