
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from dataclasses import dataclass
from invisible_core.logger import logger  # Added Logger

try:
    from pytorch_wavelets import DWTForward, DWTInverse
except ImportError:
    logger.warning("Warning: 'pytorch_wavelets' not found. DWT-Mamba will be disabled.")
    DWTForward = None
    DWTInverse = None

@dataclass
class DWTMambaConfig:
    wavelet: str = 'haar'
    levels: int = 2
    ll_weight: float = 2.0  # HARDENING: Increased from 1.0 to 2.0 to force structural changes
    hf_weight: float = 0.5  # Weight for high-freq (texture) masking

class MambaProxyBlock(nn.Module):
    """
    Simulates the effect of a Mamba SSM block using standard PyTorch ops.
    Used when the full 'mamba_ssm' CUDA kernel is not available.
    Function: Applies data-dependent non-linear distortion.
    """
    def __init__(self, d_model):
        super().__init__()
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()
    
    def forward(self, x):
        # x: [B, C, H, W] -> Flatten to [B, C, L]
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w)
        
        # SSC-like linear scan simulation
        # 1. Conv mixing
        x_conv = self.conv1d(x_flat)
        # 2. Gating/Projection
        x_gate = self.act(x_conv)
        x_out = self.proj(x_gate.transpose(1, 2)).transpose(1, 2)
        
        return x_out.view(b, c, h, w)

class DWTMambaLoss(nn.Module):
    """
    DWT-Mamba Loss Function.
    Decomposes image into frequency bands and computes loss to guide
    adversarial perturbations in specific bands (LL for structure, HH for noise).
    """
    def __init__(self, config: DWTMambaConfig, device: str = 'cuda'):
        super().__init__()
        self.config = config
        self.device = device
        
        if DWTForward is None:
            logger.warning("[DWTMambaLoss] DWTForward class is None. Disabling.")
            self.dwt = None
            self.idwt = None
            return
            
        logger.info(f"[DWTMambaLoss] Initializing DWTForward with wave={config.wavelet}")
        try:
            self.dwt = DWTForward(J=config.levels, mode='zero', wave=config.wavelet).to(device).float()
            self.idwt = DWTInverse(mode='zero', wave=config.wavelet).to(device).float()
            logger.info("[DWTMambaLoss] DWT modules created successfully.")
        except Exception as e:
            logger.error(f"[DWTMambaLoss] Failed to create DWT modules: {e}")
            self.dwt = None
            self.idwt = None
        
        # Proxy Mamba for LL band distortion
        # 3 channels (RGB)
        self.mamba_proxy = MambaProxyBlock(d_model=3).to(device)
        
    def forward(self, current_img: torch.Tensor, original_img: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DWT-Mamba Loss.
        Goal: 
        1. LL Band: Maximize distance (warp structure) OR Match distorted target
        2. HF Bands: Minimize distance (hide noise) - Optional, handled by LPIPS usually
        
        Args:
            current_img: [B, 3, H, W] - Differentiable optimization target
            original_img: [B, 3, H, W] - Reference
        """
        if self.dwt is None:
             # Return a dummy loss to allow the graph to continue without breaking
            return torch.tensor(0.0, device=self.device, requires_grad=True), {}
        
        # NaN Check for Safety
        if torch.isnan(current_img).any():
             logger.error("[DWTMambaLoss] Input contains NaN! Returning zero loss.")
             return torch.tensor(0.0, device=self.device, requires_grad=True), {}
            
        # 1. Decompose
        # Yl: Low-freq [B, 3, H_L, W_L]
        # Yh: List of High-freqs
        # FIX: pytorch_wavelets filters are Float32. Inputs from SD are BFloat16.
        # We must cast inputs to Float32 for the transform.
        curr_yl, curr_yh = self.dwt(current_img.to(dtype=torch.float32))
        orig_yl, orig_yh = self.dwt(original_img.to(dtype=torch.float32))
        
        # 2. Attack LL Band (Structure)
        # Plan: We want the current LL to be DIFFERENT from original LL
        # But random difference looks bad.
        # Strategy: Pass Original LL through Mamba Proxy to get "Warped Target"
        # and force current LL to match that warped target?
        # OR: Just Maximize L2 distance.
        
        # Better: Use Mamba to identify "structural features" independent of noise.
        # Here we maximize distance in the Mamba-transformed space?
        # Let's try: Distort Original LL -> Target LL
        # We want our image to look like the Distorted LL in frequency space.
        
        with torch.no_grad():
             # Warp the original structure
             # FIX: Ensure input matches mamba_proxy dtype (might be BFloat16)
             target_dtype = next(self.mamba_proxy.parameters()).dtype
             warped_target_ll = self.mamba_proxy(orig_yl.to(dtype=target_dtype))
             # Amplify the warp
             warped_target_ll = orig_yl + (warped_target_ll - orig_yl) * 2.0
             
        # Loss: Force current LL to match Warped LL
        loss_ll = F.mse_loss(curr_yl, warped_target_ll)
        
        # 3. Constrain High Freqs (Texture)
        # We generally want to hide changes here, LPIPS handles this.
        # But we can add explicit constraint to keep HF close to original 
        # to prevent visible artifacts while LL is warping.
        loss_hf = 0.0
        for c_h, o_h in zip(curr_yh, orig_yh):
            # c_h is [B, 3, 3, H, W] (3 orientations)
            loss_hf += F.mse_loss(c_h, o_h)
            
        total_loss = self.config.ll_weight * loss_ll + self.config.hf_weight * loss_hf
        
        return total_loss, {
            "dwt_ll": loss_ll.item(),
            "dwt_hf": loss_hf.item() if isinstance(loss_hf, torch.Tensor) else loss_hf
        }

from dataclasses import dataclass
