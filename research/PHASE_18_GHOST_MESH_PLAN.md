# Phase 18: Ghost-Mesh Protocol (Revised)

## 1. Core Architecture (Confirmed)

### A. Native PyTorch Transition
We are proceeding with `torch.nn.functional.grid_sample` (native PyTorch) instead of `kornia`. This ensures:
*   **Dependency Freedom**: No new libraries required.
*   **Gradient Flow**: Fully differentiable geometric transformations.
*   **Compatibility**: Matches Phase 17 (`protect_liquid_warp`) logic.

### B. SigLIP Advantange (Resolution Guard)
We replace `ArcFace` (112px) with **SigLIP ViT-SO400M** (384px) as the differentiable "Inner Loop" critic.
*   **Transferability**: SigLIP embeddings are robust and generalize well to black-box models.
*   **Precision**: Optimizing at 384px reduces pixelation artifacts common with 112px optimization.

### C. Gatekeeper Logic
*   **External Validator**: `Qwen-VL` runs *after* optimization.
*   **Panic Mode**: If Qwen fails (detects identity), trigger a "Retry Loop" with higher initialization noise.

---

## 2. Mathematical Model (Hinge-Loss Update)

**A. Forward Pass (Coupled Synthesis)**
$$ x_{cloaked} = \text{GridSample}(x + \delta_{noise}, \Omega_{identity} + \Delta_{warp}) $$

**B. Revised Loss Function (Hinge Constraint)**
To give the optimizer "breathing room," we use a Hinge-Loss for LPIPS. We only penalize visual degradation if it exceeds a threshold ($\tau$).

$$ L_{total} = W_{id} \cdot \text{CosSim}(E_{siglip}(x_{warped}), E_{siglip}(x_{orig})) + W_{lpips} \cdot \max(0, \text{LPIPS} - \tau) + W_{tv} \cdot TV(\delta_{noise}) $$

Where:
*   $W_{id} = 25.0$ (Primary Driver)
*   $W_{lpips} = 10.0$ (Stealth Constraint)
*   $\tau = 0.05$ (Visual Tolerance Threshold)
*   $W_{tv} = 0.01$ (Smoothness)

Effect: "I don't care how much you change the image as long as LPIPS < 0.05. Once it hits 0.05, stop and prioritize quality."

---

## 3. Implementation Plan (`src/core/protocols/ghost_mesh.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.util.ssim_loss import SSIMLoss 

class GhostMeshOpt(nn.Module):
    def __init__(self, siglip_model, device='cuda'):
        super().__init__()
        self.device = device
        self.siglip_model = siglip_model[0] # Model
        self.siglip_mean = siglip_model[1]  # Mean
        self.siglip_std = siglip_model[2]   # Std
        self.lpips_fn = None # Initialize on demand

    def forward_synthesis(self, image, noise, warp_grid):
        # 1. Add clamped noise
        noise = torch.clamp(noise, -0.05, 0.05)
        noised_img = torch.clamp(image + noise, 0, 1)

        # 2. Upsample Warp Grid (12x12 -> HxW)
        B, C, H, W = image.shape
        grid_up = F.interpolate(warp_grid.permute(0, 3, 1, 2), 
                              size=(H, W), 
                              mode='bilinear', align_corners=False)
        grid_up = grid_up.permute(0, 2, 3, 1)

        # 3. Create Identity Grid
        theta = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32, device=self.device).unsqueeze(0)
        base_grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)

        # 4. Warp
        final_flow = base_grid + grid_up
        output = F.grid_sample(noised_img, final_flow, mode='bilinear', padding_mode='reflection', align_corners=False)
        return output

    def optimize(self, image, landmarks, steps=60):
        # Optimization loop using SigLIP (384px)
        # Includes Hinge Loss logic:
        # lpips_loss = max(0, current_lpips - 0.05)
        # ...
```

## 4. Integration Strategy

1.  **Create** `src/core/protocols/ghost_mesh.py` with Hinge-Loss logic.
2.  **Update** `src/core/protocols/latent_cloak.py`:
    *   Add `protect_ghost_mesh` method.
    *   Route `target_profile="ghost_mesh"` to this method.
3.  **Verify**: Ensure SigLIP gradients flow correctly through `grid_sample`.
