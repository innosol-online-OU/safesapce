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
from typing import Tuple, Dict, Optional, Union
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
        self.w_lpips = 150.0       # Step 2: RELAXED! (Was 400.0). Prioritize Identity.
        self.w_tv = 0.05           # Minimal smoothing
        self.w_vertical = 2.0      # Keep eyes level
        self.lpips_threshold = 0.065 # Step 2: Raised from 0.045 (Allow more noise)
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
                
    def gaussian_blur(self, x, kernel_size=15, sigma=5.0):
        """Apply Gaussian Blur for 'Natural Shadow' effect."""
        k = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
        k = torch.exp(-k**2 / (2 * sigma**2))
        k = k / k.sum()
        k_2d = k[:, None] * k[None, :]
        k_2d = k_2d.expand(x.shape[1], 1, kernel_size, kernel_size)
        return F.conv2d(x, k_2d, padding=kernel_size//2, groups=x.shape[1])

    def forward_synthesis(self, image: torch.Tensor, noise: torch.Tensor, 
                          warp_grid_lr: torch.Tensor, tzone_mask: torch.Tensor,
                          face_mask: torch.Tensor = None,
                          noise_eps: float = 0.05, warp_limit: float = 0.03) -> torch.Tensor:
        """
        Coupled synthesis: Apply noise then warp.
        """
        B, C, H, W = image.shape
        
        # 1. Natural Shadows: Blur + Tanh
        # Input 'noise' is Raw. We blur it first to make it low-frequency.
        # Scaling * 10.0 restores variance lost by blur, putting Tanh in valid range.
        noise_smooth = self.gaussian_blur(noise, kernel_size=15, sigma=4.0) * 10.0
        noise_actual = torch.tanh(noise_smooth) * noise_eps
        
        # 2. Target Face (Reduce background fuzz)
        if face_mask is not None:
            # Allow 20% noise on background, 100% on face
            noise_actual = noise_actual * (0.2 + 0.8 * face_mask)
            
        noised_img = torch.clamp(image + noise_actual, 0, 1)
        
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
                               original: torch.Tensor, return_raw: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
        
        if return_raw:
            return identity_loss, cos_sim
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
    
    def generate_face_mask(self, image_tensor, face_analysis) -> torch.Tensor:
        """Broader mask covering the whole face for noise targeting."""
        B, C, H, W = image_tensor.shape
        face_mask = torch.zeros((B, 1, H, W), device=self.device)
        
        if face_analysis:
            try:
                img_np = (image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                faces = face_analysis.get(img_cv)
                if faces:
                    for face in faces:
                        x1, y1, x2, y2 = face.bbox.astype(int)
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        w_box, h_box = x2 - x1, y2 - y1
                        
                        # Ellipical mask matching bbox (Expanded for Hair)
                        # Shift center up by 15% of height to cover hair
                        cy_shifted = cy - (h_box * 0.15)
                        
                        Y, X = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
                        # Sigmoid falloff - Broader Width (0.8), Taller Height (1.2)
                        dist = ((X - cx) / (w_box * 0.8))**2 + ((Y - cy_shifted) / (h_box * 1.2))**2
                        mask_f = torch.exp(-dist * 2.0) # Sharp falloff outside face
                        face_mask[0, 0] = torch.maximum(face_mask[0, 0], mask_f)
                else:
                    face_mask.fill_(1.0) # Fallback to full image
            except:
                face_mask.fill_(1.0)
        else:
            face_mask.fill_(1.0)
            
        return face_mask

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

    def generate_semantic_grid(self, params: torch.Tensor, landmarks: torch.Tensor, 
                               grid_size: int) -> torch.Tensor:
        """
        Generate warp grid from semantic feature parameters.
        params: [eye_dist, eye_y, nose_y, mouth_y] (Tanh outputs)
        landmarks: [5, 2] normalized coordinates (0..1)
        """
        if landmarks.dim() == 3:
            landmarks = landmarks.squeeze(0)
            
        Y, X = torch.meshgrid(
            torch.linspace(0, 1, grid_size, device=self.device),
            torch.linspace(0, 1, grid_size, device=self.device),
            indexing='ij'
        )
        flow = torch.zeros(1, 2, grid_size, grid_size, device=self.device)
        
        # Radii for splats (adapt to grid)
        # Assuming typical face proportions
        r_eye = 0.005 # Squared variance
        r_mouth = 0.008
        r_nose = 0.006
        
        # 1. Eye Spacing & Height
        # 1. Eye Spacing & Height
        coord_l, coord_r = landmarks[0], landmarks[1]
        
        d_l = (X - coord_l[0])**2 + (Y - coord_l[1])**2
        mask_l = torch.exp(-d_l / r_eye)
        
        d_r = (X - coord_r[0])**2 + (Y - coord_r[1])**2
        mask_r = torch.exp(-d_r / r_eye)
        
        # Param 0: Eye Dist (Left moves -x, Right moves +x)
        # Param 1: Eye Height (Both move y)
        p_eye_w = params[0] * 0.05
        p_eye_h = params[1] * 0.05
        
        flow[0, 0] += mask_l * (-p_eye_w) + mask_r * (p_eye_w) # DX
        flow[0, 1] += (mask_l + mask_r) * p_eye_h # DY
        
        # 2. Nose Height
        nose = landmarks[2]
        d_n = (X - nose[0])**2 + (Y - nose[1])**2
        mask_n = torch.exp(-d_n / r_nose)
        
        p_nose_y = params[2] * 0.04
        flow[0, 1] += mask_n * p_nose_y
        
        # 3. Mouth Height (Smile/Frown/Open?)
        # Landmarks: 3=L_Mouth, 4=R_Mouth. Average for center.
        mouth = (landmarks[3] + landmarks[4]) / 2
        d_m = (X - mouth[0])**2 + (Y - mouth[1])**2
        mask_m = torch.exp(-d_m / r_mouth)
        
        p_mouth_y = params[3] * 0.04
        flow[0, 1] += mask_m * p_mouth_y
        
        return flow

    
    def init_attack(self, image_path: str, face_analysis=None,
                 strength: int = 75, grid_size: int = 24, num_steps: int = 60,
                 warp_noise_balance: float = 0.5, tzone_anchoring: float = 0.8,
                 tv_weight: float = 50, use_jnd: bool = True,
                 lr: float = 0.05,
                 noise_strength: float = None, warp_strength: float = None) -> Dict:
        """Initialize the attack state for granular stepping."""
        # Load image
        orig_pil = Image.open(image_path).convert('RGB')
        
        # Convert to tensor [1, 3, H, W]
        img_tensor = torch.from_numpy(np.array(orig_pil)).to(self.device).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        B, C, H, W = img_tensor.shape
        
        # Extract Semantic Landmarks
        semantic_landmarks = None
        if face_analysis:
            try:
                img_np = (img_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                faces = face_analysis.get(img_cv)
                if faces:
                    # Use first face
                    face = faces[0]
                    if hasattr(face, 'kps'):
                        # Normalize kps to 0..1
                        kps = face.kps # [5, 2]
                        semantic_landmarks = torch.from_numpy(kps).to(self.device).float()
                        semantic_landmarks[:, 0] /= W
                        semantic_landmarks[:, 1] /= H
                        logger.info("GhostMesh: Semantic Landmarks Extracted.")
            except:
                pass
        
        # Generate masks (T-Zone, Face Mask, JND)
        tzone_mask = self.generate_tzone_mask(img_tensor, face_analysis, tzone_anchoring)
        face_mask = self.generate_face_mask(img_tensor, face_analysis)
        jnd_mask = self.generate_jnd_mask(img_tensor, strength / 100.0) if use_jnd else None
        
        # Calculate effective limits based on Boosted Base (Implicitly handled by noise_eps passed later)
        if noise_strength is not None and warp_strength is not None:
             noise_eps = 0.07 * (noise_strength / 50.0)
             warp_limit = 0.05 * (warp_strength / 50.0)
        else:
             base_noise_eps = 0.07 * (strength / 50.0)
             base_warp_limit = 0.05 * (strength / 50.0)
             noise_eps = base_noise_eps * (0.3 + 0.7 * warp_noise_balance)
             warp_limit = base_warp_limit * (0.3 + 0.7 * (1 - warp_noise_balance))
             
        actual_tv_weight = tv_weight / 5000.0
        
        # Initialize Parameters
        delta_noise = torch.zeros_like(img_tensor, requires_grad=True, device=self.device)
        delta_noise.data.uniform_(-0.2, 0.2)
        
        # SEMANTIC SWITCH
        if semantic_landmarks is not None:
             # Use 4 semantic scalars: [EyeDist, EyeHeight, NoseHeight, MouthHeight]
             delta_warp = torch.zeros(4, requires_grad=True, device=self.device)
             use_semantic = True
             logger.info("GhostMesh: Using Semantic Parametric Warp.")
        else:
             # Fallback to Grid
             delta_warp = self.init_focal_bias(grid_size, scale=0.05).clone().requires_grad_(True)
             use_semantic = False
             
        optimizer = torch.optim.Adam([
            {'params': [delta_noise], 'lr': lr},
            {'params': [delta_warp],  'lr': lr * 0.1}
        ])
        
        return {
            'img_tensor': img_tensor,
            'delta_noise': delta_noise,
            'delta_warp': delta_warp, # Now might be [4] or [1, 2, G, G]
            'use_semantic': use_semantic,
            'semantic_landmarks': semantic_landmarks,
            'grid_size': grid_size,
            'tzone_mask': tzone_mask,
            'face_mask': face_mask,
            'jnd_mask': jnd_mask,
            'optimizer': optimizer,
            'metrics_history': {
                'identity_loss': [], 'lpips_raw': [], 'lpips_hinge': [],
                'tv_loss': [], 'vertical_loss': [], 'total_loss': [],
                'disp_max': [], 'cossim': [], 'step': []
            },
            'best_loss': float('inf'),
            'best_noise': None,
            'best_warp': None,
            'step': 0,
            'noise_eps': noise_eps,
            'warp_limit': warp_limit,
            'actual_tv_weight': actual_tv_weight,
            'orig_pil_size': orig_pil.size
        }
    def input_diversity(self, x):
        """
        Randomly resizes and pads the image to create 'jitter'.
        Forces structural noise (Phase 18 Hardening/DIM).
        """
        H, W = x.shape[2], x.shape[3]
        # Resize between 90% and 100%
        resize_ratio = torch.rand(1, device=self.device).item() * 0.1 + 0.9 
        h_resize = int(H * resize_ratio)
        w_resize = int(W * resize_ratio)
        
        x_resized = F.interpolate(x, size=(h_resize, w_resize), mode='bilinear', align_corners=False)
        
        # Random Padding
        pad_top = torch.randint(0, H - h_resize, (1,), device=self.device).item()
        pad_left = torch.randint(0, W - w_resize, (1,), device=self.device).item()
        pad_bottom = H - h_resize - pad_top
        pad_right = W - w_resize - pad_left
        
        return F.pad(x_resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    def train_step(self, state: Dict) -> Dict:
        """Run a single optimization step."""
        optimizer = state['optimizer']
        optimizer.zero_grad()
        
        delta_noise = state['delta_noise']
        delta_warp_param = state['delta_warp']
        
        # Generate Warp Grid (Parametric or Dense)
        if state.get('use_semantic', False):
             # Parametric -> Grid
             # Tanh constraint on parameters
             params_constrained = torch.tanh(delta_warp_param)
             delta_warp = self.generate_semantic_grid(params_constrained, state['semantic_landmarks'], state['grid_size'])
             # Apply global scale
             delta_warp = delta_warp * state['warp_limit']
        else:
             # Dense Grid
             delta_warp = delta_warp_param
        
        # Apply JND
        if state.get('jnd_mask') is not None:
             noise_masked = delta_noise * state['jnd_mask']
        else:
             noise_masked = delta_noise
             
        # Forward
        warped = self.forward_synthesis(
            state['img_tensor'], noise_masked, delta_warp, state['tzone_mask'],
            face_mask=state.get('face_mask'),
            noise_eps=state['noise_eps'], warp_limit=state['warp_limit']
        )
        
        # Phase 18 Hardening: Input Diversity (Forces structural features)
        warped_diverse = self.input_diversity(warped)
        
        # Loss (Train on Diverse, Log on Clean)
        identity_loss, _ = self.compute_identity_loss(warped_diverse, state['img_tensor'], return_raw=True)
        
        # Compute clean similarity for logging (so graph isn't jittery)
        with torch.no_grad():
             _, cossim_val = self.compute_identity_loss(warped, state['img_tensor'], return_raw=True)
        
        lpips_hinge, lpips_raw = self.compute_lpips_hinge_loss(warped, state['img_tensor'])
        
        # Regularize ACTUAL perturbation (not latent)
        actual_noise = torch.tanh(noise_masked) * state['noise_eps']
        actual_warp = torch.tanh(delta_warp) * state['warp_limit']
        
        tv_loss = self.compute_tv_loss(actual_noise)
        vertical_loss = self.compute_vertical_loss(actual_warp)
        
        total_loss = (self.w_identity * identity_loss + 
                      self.w_lpips * lpips_hinge +
                      state['actual_tv_weight'] * tv_loss +
                      self.w_vertical * vertical_loss)
                      
        total_loss.backward()
        optimizer.step()
        
        # Track
        disp_max = delta_warp.abs().max().item()
        step = state['step']
        state['metrics_history']['step'].append(step)
        state['metrics_history']['identity_loss'].append(identity_loss.item())
        state['metrics_history']['lpips_raw'].append(lpips_raw.item())
        state['metrics_history']['lpips_hinge'].append(lpips_hinge.item())
        state['metrics_history']['tv_loss'].append(tv_loss.item())
        state['metrics_history']['vertical_loss'].append(vertical_loss.item())
        state['metrics_history']['total_loss'].append(total_loss.item())
        state['metrics_history']['disp_max'].append(disp_max)
        state['metrics_history']['cossim'].append(cossim_val.item())
        
        # Best
        if total_loss.item() < state['best_loss']:
            state['best_loss'] = total_loss.item()
            state['best_noise'] = delta_noise.data.clone()
            state['best_warp'] = delta_warp.data.clone()
            
        state['step'] += 1
        return state

    def get_current_image(self, state: Dict, use_best: bool = False) -> Image.Image:
        """Generate PIL image from current or best state."""
        with torch.no_grad():
            if use_best:
                noise = state['best_noise']
                warp = state['best_warp']
            else:
                noise = state['delta_noise']
                warp = state['delta_warp']
                
            # Semantic Warp Conversion (Fix for Shape Mismatch)
            if state.get('use_semantic', False):
                 # Parametric -> Grid
                 params_constrained = torch.tanh(warp)
                 warp = self.generate_semantic_grid(params_constrained, state['semantic_landmarks'], state['grid_size'])
                 warp = warp * state['warp_limit']
                
            if state.get('jnd_mask') is not None:
                noise = noise * state['jnd_mask']
                
            warped = self.forward_synthesis(
                state['img_tensor'], noise, warp, state['tzone_mask'],
                noise_eps=state['noise_eps'], warp_limit=state['warp_limit']
            )
            
            # Denormalize
            warped_np = warped.squeeze(0).permute(1, 2, 0).cpu().numpy()
            warped_np = (warped_np * 255.0).clip(0, 255).astype(np.uint8)
            return Image.fromarray(warped_np)

    def optimize(self, image_path: str, face_analysis=None,
                 strength: int = 75, grid_size: int = 24, num_steps: int = 60,
                 warp_noise_balance: float = 0.5, tzone_anchoring: float = 0.8,
                 tv_weight: float = 50, use_jnd: bool = True,
                 lr: float = 0.05,
                 noise_strength: float = None, warp_strength: float = None) -> Tuple[Image.Image, Dict]:
        """
        Run Ghost-Mesh optimization loop (Legacy Wrapper).
        """
        state = self.init_attack(image_path, face_analysis, strength, grid_size, num_steps,
                                 warp_noise_balance, tzone_anchoring, tv_weight, use_jnd, lr,
                                 noise_strength, warp_strength)
        
        logger.info(f"[GhostMesh] Starting optimization: Steps={num_steps}...")
        
        for step in range(num_steps):
            state = self.train_step(state)
            
            if step % 10 == 0:
                 # Minimal logging
                 metrics = state['metrics_history']
                 id_loss = metrics['identity_loss'][-1]
                 lpips = metrics['lpips_hinge'][-1]
                 print(f"GhostMesh Step {step}/{num_steps} | ID: {id_loss:.4f} | LPIPS: {lpips:.4f}", flush=True)
                 
        # Final result (Best)
        result_pil = self.get_current_image(state, use_best=True)
        return result_pil, state['metrics_history']
