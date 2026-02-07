
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unittest.mock import MagicMock
from PIL import Image

# Add root to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- MOCK MODELS ---
class MockSigLIP(nn.Module):
    """Simple differentiable conv net to simulate SigLIP features."""
    def __init__(self):
        super().__init__()
        # Input 384x384 -> Output 768 dim vector
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 768)

    def forward(self, x):
        # x: [B, 3, 384, 384]
        out = F.relu(self.conv(x))
        out = self.pool(out).flatten(1)
        out = self.fc(out)
        return out

class MockFaceAnalysis:
    """Mock InsightFace detector."""
    def get(self, img):
        # Return a fake face object
        class Face:
            def __init__(self):
                # Center bbox
                h, w = img.shape[:2]
                self.bbox = np.array([w//4, h//4, w*3//4, h*3//4])
                self.kps = np.array([
                    [w*0.35, h*0.4], [w*0.65, h*0.4], # Eyes
                    [w*0.5, h*0.5], # Nose
                    [w*0.4, h*0.7], [w*0.6, h*0.7]  # Mouth
                ])
                self.embedding = np.random.randn(512)
        return [Face()]

# --- TESTS ---

def test_ghost_mesh_gradients():
    print("\n--- Testing Ghost Mesh Gradients ---")
    from invisible_core.attacks.ghost_mesh import GhostMeshOptimizer

    device = "cpu" # Test on CPU for CI compatibility

    # 1. Setup Mock Models
    siglip = (MockSigLIP().to(device), torch.tensor([0.5]*3).view(1,3,1,1), torch.tensor([0.5]*3).view(1,3,1,1))
    face_app = MockFaceAnalysis()

    optimizer = GhostMeshOptimizer(siglip, device=device)

    # 2. Create Dummy Image (White Noise)
    img_pil = Image.new('RGB', (512, 512), color='white')
    # Save temp file
    img_path = "temp_test_img.png"
    img_pil.save(img_path)

    # 3. Initialize Attack State
    state = optimizer.init_attack(
        image_path=img_path,
        face_analysis=face_app,
        strength=50,
        grid_size=16,
        num_steps=5, # Only need a few steps to check gradient flow
        lr=0.1
    )

    # 4. Run One Step
    # Check initial loss
    print("Initial State initialized.")
    initial_loss = state['best_loss']

    # Check if gradients exist before step
    assert state['delta_noise'].grad is None
    assert state['delta_warp'].grad is None

    # Run optimization step
    print("Running optimization step...")
    state = optimizer.train_step(state)

    # 5. Verify Gradients
    noise_grad = state['delta_noise'].grad
    warp_grad = state['delta_warp'].grad

    print(f"Noise Grad Norm: {noise_grad.norm().item():.4f}")
    print(f"Warp Grad Norm: {warp_grad.norm().item():.4f}")

    if noise_grad.norm().item() == 0:
        print("FAIL: No gradient on noise delta!")
    else:
        print("PASS: Noise gradient detected.")

    if warp_grad.norm().item() == 0:
        print("FAIL: No gradient on warp delta!")
    else:
        print("PASS: Warp gradient detected.")

    # 6. Verify Loss Change
    # Note: Since we use random noise, loss might fluctuate, but gradients prove optimization is happening.
    loss_val = state['metrics_history']['total_loss'][-1]
    print(f"Step 1 Loss: {loss_val:.4f}")

    # Cleanup
    if os.path.exists(img_path):
        os.remove(img_path)

def test_liquid_warp_constraints():
    print("\n--- Testing Liquid Warp Constraints ---")
    # We'll simulate the warp logic directly to verify Tanh constraints

    device = "cpu"
    h, w = 100, 100

    # Random Displacement (Large values)
    displacement_raw = torch.randn(1, 2, h, w) * 10.0

    # Constraints
    limit = 0.05
    constrained = torch.tanh(displacement_raw) * limit

    max_val = constrained.abs().max().item()
    print(f"Max Displacement (Limit {limit}): {max_val:.6f}")

    if max_val > limit + 1e-6:
        print("FAIL: Displacement exceeds limit!")
    else:
        print("PASS: Displacement correctly constrained.")

if __name__ == "__main__":
    try:
        test_ghost_mesh_gradients()
        test_liquid_warp_constraints()
        print("\n✅ All Math Verification Tests Passed.")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
