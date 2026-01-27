"""
Phase 18: Ghost-Mesh Protocol
Coupled Warp + Noise Optimization with Hinge-Loss Constraints.

This module combines Phase 16 (Resonant Ghost) pixel perturbation with
Phase 17 (Liquid Warp) geometric warping into a unified optimization loop.

Key Features:
- Joint optimization of noise (delta) and warp (displacement field)
- SigLIP (ViT-SO400M, 384px) as the differentiable critic
- Hinge-Loss LPIPS constraint for visual tolerance
- T-Zone anchoring to preserve silhouette
- JND masking for perceptual invisibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict, Optional
import os

from src.core.logger import logger


class GhostMeshOptimizer(nn.Module):
    """
    Ghost-Mesh Optimizer: Coupled Warp + Noise Attack.
    
    Mathematical Model:
        x_cloaked = GridSample(x + delta_noise, identity_grid + delta_warp)
        
    Loss Function (Hinge-Loss):
        L_total = W_id * CosSim(E_siglip(x_warped), E_siglip(x_orig))
                + W_lpips * max(0, LPIPS - tau)
                + W_tv * TV(delta_noise)
                + W_vert * |delta_warp_y|
                
    Where:
        W_id = 25.0 (Primary Driver)
        W_lpips = 10.0 (Stealth Constraint)
        tau = 0.05 (Visual Tolerance Threshold)
        W_tv = 0.01 (Smoothness)
        W_vert = 10.0 (Vertical Penalty)
    """
    
    def __init__(self, siglip_model, device='cuda'):
        """
        Initialize Ghost-Mesh Optimizer.
        
        Args:
            siglip_model: Tuple of (model, mean, std) for SigLIP
            device: 'cuda' or 'cpu'
        """
        super().__init__()
        self.device = device
        self.siglip_model = siglip_model[0]  # Model
        self.siglip_mean = siglip_model[1]   # Mean normalization
        self.siglip_std = siglip_model[2]    # Std normalization
        self.lpips_fn = None  # Lazy-loaded LPIPS
        
        # Loss weights (Phase 18 CORRECTED - Scale Matched)
        self.w_identity = 300.0    # Step 1: Push hard until 0.25 (Blind the critic)
        self.w_lpips = 400.0       # Step 2: Looser brake (Allow 0.045 texture change)
        self.w_tv = 0.05           # Minimal smoothing
        self.w_vertical = 2.0      # Keep eyes level
        self.lpips_threshold = 0.045 # Step 2: Raised from 0.02 to allow semantic noise
        self.identity_threshold = 0.25 # Step 1: Lowered from 0.5 to force deeper attack
        
    def _load_lpips(self):
        """Lazy-load LPIPS metric."""
        if self.lpips_fn is None:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
                self.lpips_fn.eval()
                for param in self.lpips_fn.parameters():
                    param.requires_grad = False
                logger.info("[GhostMesh] LPIPS (alex) loaded.")
            except ImportError:
                logger.warning("[GhostMesh] LPIPS not available. Falling back to L1 loss.")
                self.lpips_fn = None
                
    def forward_synthesis(self, image: torch.Tensor, noise: torch.Tensor, 
                          warp_grid_lr: torch.Tensor, tzone_mask: torch.Tensor,
                          noise_eps: float = 0.05, warp_limit: float = 0.03) -> torch.Tensor:
        """
        Coupled synthesis: Apply noise then warp.
        
        Args:
            image: [B, 3, H, W] original image tensor in [0, 1]
            noise: [B, 3, H, W] perturbation delta
            warp_grid_lr: [B, 2, G, G] low-res displacement field
            tzone_mask: [B, 1, H, W] T-zone anchoring mask
            noise_eps: Max noise magnitude
            warp_limit: Max warp displacement (normalized)
            
        Returns:
            [B, 3, H, W] cloaked image
        """
        B, C, H, W = image.shape
        
        # 1. Clamp and apply noise
        noise_clamped = torch.clamp(noise, -noise_eps, noise_eps)
        noised_img = torch.clamp(image + noise_clamped, 0, 1)
        
        # 2. Upsample low-res warp grid to full resolution
        # warp_grid_lr: [B, 2, G, G] -> [B, 2, H, W]
        warp_upsampled = F.interpolate(warp_grid_lr, size=(H, W), 
                                        mode='bicubic', align_corners=False)
        
        # 3. Apply Tanh constraint and T-Zone mask
        warp_constrained = torch.tanh(warp_upsampled) * warp_limit
        warp_masked = warp_constrained * tzone_mask
        
        # 4. Create identity grid + displacement
        theta = torch.tensor([[1, 0, 0], [0, 1, 0]], 
                             dtype=torch.float32, device=self.device)
        theta = theta.unsqueeze(0).expand(B, -1, -1)
        identity_grid = F.affine_grid(theta, (B, C, H, W), align_corners=True)
        
        # 5. Add warp displacement (permute from [B, 2, H, W] to [B, H, W, 2])
        final_grid = identity_grid + warp_masked.permute(0, 2, 3, 1)
        
        # 6. Warp the noised image
        output = F.grid_sample(noised_img, final_grid, mode='bilinear',
                               padding_mode='border', align_corners=True)
        
        return output
    
    def compute_identity_loss(self, warped: torch.Tensor, 
                               original: torch.Tensor) -> torch.Tensor:
        """
        Compute Hinge-Based Identity Loss using SigLIP.
        
        Uses ReLU(CosSim - threshold) to create a target-based loss.
        Once CosSim drops below threshold, loss becomes 0 (stop pushing).
        
        Args:
            warped: [B, 3, H, W] warped image
            original: [B, 3, H, W] original image
            
        Returns:
            Scalar loss (0 when identity is broken, positive otherwise)
        """
        # Resize to SigLIP input size (384x384)
        warped_384 = F.interpolate(warped, size=(384, 384), mode='bilinear')
        orig_384 = F.interpolate(original, size=(384, 384), mode='bilinear')
        
        # Normalize
        warped_norm = (warped_384 - self.siglip_mean) / self.siglip_std
        orig_norm = (orig_384 - self.siglip_mean) / self.siglip_std
        
        # Extract features
        with torch.no_grad():
            orig_features = self.siglip_model(orig_norm)
            orig_features = orig_features / orig_features.norm(dim=-1, keepdim=True)
        
        warped_features = self.siglip_model(warped_norm)
        warped_features = warped_features / warped_features.norm(dim=-1, keepdim=True)
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(warped_features, orig_features).mean()
        
        # FIX: Hinge-loss with target threshold
        # Returns 0 when CosSim < threshold (identity broken!)
        # Returns positive when CosSim > threshold (keep pushing)
        identity_loss = F.relu(cos_sim - self.identity_threshold)
        
        return identity_loss
    
    def compute_lpips_hinge_loss(self, warped: torch.Tensor, 
                                  original: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Hinge-Loss LPIPS: max(0, LPIPS - tau).
        
        Only penalizes if visual degradation exceeds threshold.
        
        Args:
            warped: [B, 3, H, W] warped image in [0, 1]
            original: [B, 3, H, W] original image in [0, 1]
            
        Returns:
            Tuple of (hinge_loss, raw_lpips)
        """
        self._load_lpips()
        
        if self.lpips_fn is not None:
            # LPIPS expects [-1, 1] range
            warped_scaled = warped * 2 - 1
            orig_scaled = original * 2 - 1
            
            lpips_val = self.lpips_fn(warped_scaled, orig_scaled).mean()
            hinge_loss = F.relu(lpips_val - self.lpips_threshold)
        else:
            # Fallback: L1 loss with threshold
            lpips_val = torch.abs(warped - original).mean()
            hinge_loss = F.relu(lpips_val - self.lpips_threshold)
            
        return hinge_loss, lpips_val
    
    def compute_tv_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Total Variation loss for smoothness.
        
        Args:
            x: [B, C, H, W] tensor
            
        Returns:
            Scalar TV loss
        """
        tv_h = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        tv_v = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return tv_h + tv_v
    
    def compute_vertical_loss(self, warp_grid: torch.Tensor) -> torch.Tensor:
        """
        Vertical penalty to keep eyes level.
        
        Args:
            warp_grid: [B, 2, G, G] displacement field
            
        Returns:
            Scalar penalty for vertical movement
        """
        # Channel 1 = dy (vertical displacement)
        return torch.mean(torch.abs(warp_grid[:, 1, :, :]))
    
    def generate_tzone_mask(self, image_tensor: torch.Tensor, 
                            face_analysis=None, anchoring_strength: float = 0.8) -> torch.Tensor:
        """
        Generate T-Zone mask for anchored warping.
        
        T-Zone = inner facial features (eyes, nose, mouth)
        Silhouette = outer boundary (jawline, hair) - frozen
        
        Args:
            image_tensor: [B, 3, H, W] image tensor
            face_analysis: InsightFace FaceAnalysis object
            anchoring_strength: 0.0 = full warp, 1.0 = freeze jawline
            
        Returns:
            [B, 1, H, W] mask where 1.0 = warp, 0.0 = frozen
        """
        B, C, H, W = image_tensor.shape
        tzone_mask = torch.zeros((B, 1, H, W), device=self.device)
        
        if face_analysis is None:
            # Fallback: center-weighted Gaussian
            cy, cx = H // 2, W // 2
            radius = min(H, W) * 0.3
            
            Y, X = torch.meshgrid(
                torch.arange(H, device=self.device),
                torch.arange(W, device=self.device),
                indexing='ij'
            )
            dist = ((Y - cy)**2 + (X - cx)**2).sqrt()
            tzone_mask[0, 0] = torch.exp(-(dist / max(radius, 1))**2)
            logger.info("[GhostMesh] Using center fallback T-Zone mask.")
        else:
            # Use InsightFace landmarks
            try:
                img_np = (image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                faces = face_analysis.get(img_cv)
                
                if faces:
                    for face in faces:
                        if hasattr(face, 'kps') and face.kps is not None:
                            kps = face.kps  # [L_eye, R_eye, Nose, L_mouth, R_mouth]
                            l_eye, r_eye, nose = kps[0], kps[1], kps[2]
                            
                            # T-zone center
                            center_x = (l_eye[0] + r_eye[0]) / 2
                            center_y = (l_eye[1] + nose[1]) / 2
                            
                            # Ellipse radii
                            radius_x = abs(r_eye[0] - l_eye[0]) * 0.8
                            radius_y = abs(nose[1] - l_eye[1]) * 1.5
                            
                            Y, X = torch.meshgrid(
                                torch.arange(H, device=self.device),
                                torch.arange(W, device=self.device),
                                indexing='ij'
                            )
                            ellipse_dist = ((X - center_x) / max(radius_x, 1))**2 + \
                                           ((Y - center_y) / max(radius_y, 1))**2
                            face_tzone = torch.exp(-ellipse_dist * 0.5)
                            tzone_mask[0, 0] = torch.maximum(tzone_mask[0, 0], face_tzone)
                        else:
                            # Fallback to bbox center
                            x1, y1, x2, y2 = face.bbox.astype(int)
                            cx, cy = (x1 + x2) / 2, y1 + (y2 - y1) * 0.35
                            radius = min(x2 - x1, y2 - y1) * 0.4
                            
                            Y, X = torch.meshgrid(
                                torch.arange(H, device=self.device),
                                torch.arange(W, device=self.device),
                                indexing='ij'
                            )
                            dist = ((Y - cy)**2 + (X - cx)**2).sqrt()
                            face_tzone = torch.exp(-(dist / max(radius, 1))**2)
                            tzone_mask[0, 0] = torch.maximum(tzone_mask[0, 0], face_tzone)
                    
                    logger.info(f"[GhostMesh] T-Zone mask generated from {len(faces)} face(s).")
                else:
                    # No face detected - use center fallback
                    cy, cx = H // 2, W // 2
                    radius = min(H, W) * 0.3
                    Y, X = torch.meshgrid(
                        torch.arange(H, device=self.device),
                        torch.arange(W, device=self.device),
                        indexing='ij'
                    )
                    dist = ((Y - cy)**2 + (X - cx)**2).sqrt()
                    tzone_mask[0, 0] = torch.exp(-(dist / max(radius, 1))**2)
                    logger.warning("[GhostMesh] No face detected. Using center fallback.")
                    
            except Exception as e:
                logger.error(f"[GhostMesh] Face detection failed: {e}")
                # Fallback
                tzone_mask.fill_(0.5)
        
        # Apply anchoring strength (higher = more frozen at edges)
        # FIX: Higher anchoring = SMALLER mask = LESS warp on critical areas
        if anchoring_strength > 0:
            # Raise mask to higher power = shrinks active area
            tzone_mask = tzone_mask ** (1.0 + anchoring_strength * 2.0)
        
        # Smooth the mask
        mask_np = tzone_mask[0, 0].cpu().numpy()
        mask_np = cv2.GaussianBlur(mask_np, (15, 15), 5)
        tzone_mask = torch.from_numpy(mask_np).to(self.device).unsqueeze(0).unsqueeze(0)
        
        return tzone_mask
    
    def generate_jnd_mask(self, image_tensor: torch.Tensor, 
                          strength: float = 1.0) -> torch.Tensor:
        """
        Generate Just Noticeable Difference mask for noise.
        
        Higher values in textured areas, lower in smooth areas.
        
        Args:
            image_tensor: [B, 3, H, W] image in [0, 1]
            strength: Multiplier for the mask
            
        Returns:
            [B, 3, H, W] JND mask
        """
        B, C, H, W = image_tensor.shape
        
        # Convert to grayscale for edge detection
        gray = 0.299 * image_tensor[:, 0:1] + 0.587 * image_tensor[:, 1:2] + 0.114 * image_tensor[:, 2:3]
        
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        edges_x = F.conv2d(gray, sobel_x, padding=1)
        edges_y = F.conv2d(gray, sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2 + 1e-8)
        
        # Normalize and expand to 3 channels
        edges = edges / (edges.max() + 1e-8)
        jnd_mask = edges.expand(-1, 3, -1, -1) * strength
        
        # Base floor for smooth areas
        jnd_mask = torch.clamp(jnd_mask + 0.02, 0.02, 0.15)
        
        return jnd_mask
    
    def init_focal_bias(self, grid_size: int, scale: float = 0.05) -> torch.Tensor:
        """
        Initialize displacement field with focal length bias.
        
        Simulates wide-angle lens distortion: horizontal expansion at center.
        
        Args:
            grid_size: Size of the displacement grid
            scale: Intensity of focal bias
            
        Returns:
            [1, 2, G, G] initial displacement field
        """
        y = torch.linspace(-1, 1, grid_size, device=self.device).view(grid_size, 1)
        x = torch.linspace(-1, 1, grid_size, device=self.device).view(1, grid_size)
        r2 = (x**2 + y**2).clamp(0, 1).unsqueeze(0).unsqueeze(0)
        
        # Focal weight: (1 - r^2) -> max at center
        focal_weight = 1.0 - r2
        
        # Horizontal expansion, no vertical movement
        dx = x.unsqueeze(0).unsqueeze(0) * scale * focal_weight
        dy = torch.zeros_like(dx)
        
        return torch.cat([dx, dy], dim=1)
    
    def optimize(self, image_path: str, face_analysis=None,
                 strength: int = 75, grid_size: int = 24, num_steps: int = 60,
                 warp_noise_balance: float = 0.5, tzone_anchoring: float = 0.8,
                 tv_weight: float = 50, use_jnd: bool = True,
                 lr: float = 0.05,
                 noise_strength: float = None, warp_strength: float = None) -> Tuple[Image.Image, Dict]:
        """
        Run Ghost-Mesh optimization loop.
        
        Args:
            image_path: Path to input image
            face_analysis: InsightFace FaceAnalysis for landmark detection
            strength: Attack intensity (0-100)
            grid_size: Low-res displacement grid size
            num_steps: Optimization iterations
            warp_noise_balance: 0.0 = warp-heavy, 1.0 = noise-heavy
            tzone_anchoring: 0.0 = full warp, 1.0 = freeze silhouette
            tv_weight: Total Variation weight (UI slider 1-100)
            use_jnd: Enable JND masking for noise
            lr: Learning rate for Adam optimizer
            
        Returns:
            (protected_image, metrics_history)
        """
        # Load image
        orig_pil = Image.open(image_path).convert('RGB')
        w_orig, h_orig = orig_pil.size
        
        # Convert to tensor [1, 3, H, W]
        img_tensor = torch.from_numpy(np.array(orig_pil)).to(self.device).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Generate masks
        tzone_mask = self.generate_tzone_mask(img_tensor, face_analysis, tzone_anchoring)
        jnd_mask = self.generate_jnd_mask(img_tensor, strength / 100.0) if use_jnd else None
        
        # Calculate effective limits
        # If granular controls are provided, use them directly
        if noise_strength is not None and warp_strength is not None:
             # Direct mapping: 50 -> 0.05 noise, 50 -> 0.03 warp
             noise_eps = 0.05 * (noise_strength / 50.0)
             warp_limit = 0.03 * (warp_strength / 50.0)
        else:
             # Legacy Strength/Balance logic
             base_noise_eps = 0.05 * (strength / 50.0)
             base_warp_limit = 0.03 * (strength / 50.0)
             
             noise_eps = base_noise_eps * (0.3 + 0.7 * warp_noise_balance)
             warp_limit = base_warp_limit * (0.3 + 0.7 * (1 - warp_noise_balance))
        
        # Map TV weight from UI (1-100) to actual weight
        actual_tv_weight = tv_weight / 5000.0  # 50 -> 0.01
        
        # Initialize learnable parameters
        # FIX: Init with small random noise to kickstart optimization
        delta_noise = torch.zeros_like(img_tensor, requires_grad=True, device=self.device)
        delta_noise.data.uniform_(-0.01, 0.01)
        delta_warp = self.init_focal_bias(grid_size, scale=0.05).clone().requires_grad_(True)
        
        # Step 3: Split optimizer learning rates (Noise First)
        optimizer = torch.optim.Adam([
            {'params': [delta_noise], 'lr': 0.05},   # Aggressive Noise
            {'params': [delta_warp],  'lr': 0.005}   # Subtle Warp (1/10th strength)
        ])
        
        # Metrics tracking
        metrics_history = {
            'step': [],
            'identity_loss': [],
            'lpips_raw': [],      # Raw LPIPS for visualization
            'lpips_hinge': [],    # Hinge loss used in optimization
            'tv_loss': [],
            'vertical_loss': [],
            'total_loss': [],
            'disp_max': []
        }
        
        best_loss = float('inf')
        best_noise = delta_noise.data.clone()
        best_warp = delta_warp.data.clone()
        
        logger.info(f"[GhostMesh] Starting optimization: Steps={num_steps}, Grid={grid_size}x{grid_size}, "
                    f"Balance={warp_noise_balance:.2f}, Anchoring={tzone_anchoring:.2f}")
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Apply JND mask to noise if enabled
            if use_jnd and jnd_mask is not None:
                noise_masked = delta_noise * jnd_mask
            else:
                noise_masked = delta_noise
            
            # Forward synthesis
            warped = self.forward_synthesis(
                img_tensor, noise_masked, delta_warp, tzone_mask,
                noise_eps=noise_eps, warp_limit=warp_limit
            )
            
            # Compute losses
            identity_loss = self.compute_identity_loss(warped, img_tensor)
            lpips_hinge, lpips_raw = self.compute_lpips_hinge_loss(warped, img_tensor)
            tv_loss = self.compute_tv_loss(delta_noise)
            vertical_loss = self.compute_vertical_loss(delta_warp)
            
            # Total loss with weights
            total_loss = (self.w_identity * identity_loss + 
                          self.w_lpips * lpips_hinge +
                          actual_tv_weight * tv_loss +
                          self.w_vertical * vertical_loss)
            
            total_loss.backward()
            optimizer.step()
            
            # Track metrics
            disp_max = delta_warp.abs().max().item()
            metrics_history['step'].append(step)
            metrics_history['identity_loss'].append(identity_loss.item())
            metrics_history['lpips_raw'].append(lpips_raw.item())      # Raw LPIPS
            metrics_history['lpips_hinge'].append(lpips_hinge.item())  # Hinge loss
            metrics_history['tv_loss'].append(tv_loss.item())
            metrics_history['vertical_loss'].append(vertical_loss.item())
            metrics_history['total_loss'].append(total_loss.item())
            metrics_history['disp_max'].append(disp_max)
            
            # Track best
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_noise = delta_noise.data.clone()
                best_warp = delta_warp.data.clone()
            
            if step % 10 == 0:
                print(f"GhostMesh Step {step}/{num_steps} | ID: {identity_loss.item():.4f} | "
                      f"LPIPS(raw): {lpips_raw.item():.4f} | Hinge: {lpips_hinge.item():.4f} | "
                      f"TV: {tv_loss.item():.4f} | Vert: {vertical_loss.item():.4f} | DispMax: {disp_max:.6f}", flush=True)
        
        # Final output with best parameters
        with torch.no_grad():
            if use_jnd and jnd_mask is not None:
                noise_masked = best_noise * jnd_mask
            else:
                noise_masked = best_noise
                
            final_warped = self.forward_synthesis(
                img_tensor, noise_masked, best_warp, tzone_mask,
                noise_eps=noise_eps, warp_limit=warp_limit
            )
        
        # Convert to PIL
        final_np = final_warped.cpu().permute(0, 2, 3, 1).numpy()[0]
        final_pil = Image.fromarray((final_np * 255).astype(np.uint8))
        
        logger.info(f"[GhostMesh] Complete. Best Loss: {best_loss:.4f}")
        
        return final_pil, metrics_history
