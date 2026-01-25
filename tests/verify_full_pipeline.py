import torch
import numpy as np
from PIL import Image
import sys
import os

# Ensure src is in path
sys.path.append(os.getcwd())
from invisible_core.attacks.ghost_mesh import GhostMeshOptimizer

class MockSigLIP:
    def __init__(self):
        self.device = 'cpu' # Used by code checking model.device
        self.config = type('cfg', (), {'hidden_size': 768})()
        
    def __call__(self, pixel_values):
        # Return dummy features [B, 768] (Tensor)
        return torch.randn(1, 768)

    def to(self, device):
        return self

class MockProcessor:
    class MockImgProc:
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.5, 0.5, 0.5]
    image_processor = MockImgProc()
    
    def __call__(self, images=None, return_tensors="pt"):
        return {'pixel_values': torch.randn(1, 3, 224, 224)}

def test_pipeline():
    print("\nStarting Full Pipeline Verification...")
    
    # 1. Create Dummy Image
    if not os.path.exists("tests"): os.makedirs("tests")
    img_path = "tests/dummy_face.png"
    Image.new('RGB', (512, 512), color=(128,128,128)).save(img_path)
    
    # 2. Init Optimizer
    # Inject Mock SigLIP (Model, Processor, Std)
    mock_model = MockSigLIP()
    mock_processor = MockProcessor() # Not needed in tuple
    mean_tensor = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std_tensor = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    opt = GhostMeshOptimizer(siglip_model=(mock_model, mean_tensor, std_tensor))
    opt.device = torch.device('cpu') # Force CPU execution
    
    # 3. Init Attack
    print("\n[Init Phase]")
    try:
        state = opt.init_attack(img_path, face_analysis=None, strength=50)
        print("init_attack success.")
    except Exception as e:
        print(f"init_attack failed: {e}")
        return

    # Check Keys
    required = ['identity_loss', 'step', 'best_loss']
    missing = [k for k in required if k not in state.get('metrics_history', {}) and k not in state]
    if missing:
        print(f"Missing keys in state: {missing}")
        return
            
    # 4. Train Steps
    print("\n[Training Phase]")
    for i in range(5):
        try:
            state = opt.train_step(state)
            loss = state['metrics_history']['total_loss'][-1]
            step = state['metrics_history']['step'][-1]
            print(f"  Step {step}: Success. Loss={loss:.4f}")
        except Exception as e:
            print(f"Crash at Step {i+1}: {e}")
            import traceback
            traceback.print_exc()
            return

    # 5. Visualization check
    print("\n[Visualization Phase]")
    try:
        img = opt.get_current_image(state)
        print(f"  get_current_image: OK ({img.size})")
    except Exception as e:
         print(f"get_current_image failed: {e}")
         return
    
    try:
        # Simulate SessionManager get_components behavior
        params = state['delta_warp']
        # If semantic, convert
        if state.get('use_semantic', False):
             p = torch.tanh(params)
             w = opt.generate_semantic_grid(p, state['semantic_landmarks'], state['grid_size'])
        else:
             w = params
        mag = (w**2).sum(dim=1).sqrt().sum().item()
        print(f"  get_components logic: OK (Mag={mag:.2f})")
    except Exception as e:
         print(f"get_components simulation failed: {e}")
         return

    print("\nVerification COMPLETE. Pipeline is stable.")
    
if __name__ == "__main__":
    test_pipeline()
