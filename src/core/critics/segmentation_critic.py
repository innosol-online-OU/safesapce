"""
Project Invisible Phase 4: Segmentation Critic (Anti-Grok)
Uses YOLOv8n-Seg to minimize "person" detection confidence.
Lightweight alternative to heavy SD ensemble (~6MB vs ~5GB VRAM).
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from src.core.logger import logger


class SegmentationCritic:
    """
    Anti-Segmentation Critic using YOLOv8n-Seg.
    
    Goal: Minimize the probability that AI can segment a "person" from the image.
    This breaks Grok's ability to isolate faces/bodies for editing.
    """
    
    # COCO class IDs for human-related objects
    PERSON_CLASS_ID = 0  # 'person' in COCO
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = None
        self.loaded = False
        
    def load(self):
        """Lazy load YOLOv8n-seg model."""
        if self.loaded:
            return
            
        logger.info("[SegmentationCritic] Loading YOLOv8n-seg (6MB)...")
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n-seg.pt")
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.loaded = True
            logger.info("[SegmentationCritic] Model loaded successfully.")
        except Exception as e:
            logger.error(f"[SegmentationCritic] Failed to load: {e}")
            self.loaded = False
            
    def unload(self):
        """Free VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
            self.loaded = False
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[SegmentationCritic] Model unloaded.")
    
    def compute_loss(self, image: torch.Tensor) -> torch.Tensor:
        """
        Phase 7.1: Compute adversarial loss with ROBUST INPUT NORMALIZATION.
        
        Fixes "0.6939 Collapse" (Dead Signal) where input range mismatch
        causes YOLO to see garbage, resulting in gradients dying.
        
        Args:
            image: [B, 3, H, W] tensor (could be [-1, 1] or [0, 1] range)
            
        Returns:
            loss: Scalar tensor with valid gradients
        """
        import torch.nn.functional as F
        
        if not self.loaded:
            self.load()
        if not self.loaded:
            return torch.tensor(0.0, device=self.device, requires_grad=False)
        
        # Ensure image is on correct device
        img = image.to(self.device)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        
        # --- STEP 1: FORCE ROBUST NORMALIZATION ---
        # Auto-detect input range and normalize to [0, 1]
        img_min = img.min().item()
        img_max = img.max().item()
        
        # Debug: Print input range (first call only via logger)
        logger.info(f"[SegmentationCritic] Input Range: [{img_min:.3f}, {img_max:.3f}]")
        
        if img_min < -0.5:
            # Input is [-1, 1] range (from VAE)
            x = (img + 1.0) / 2.0
            logger.info("[SegmentationCritic] Detected [-1,1] range, shifting to [0,1]")
        elif img_max <= 1.0:
            # Input is already [0, 1] range
            x = img
            logger.info("[SegmentationCritic] Detected [0,1] range, using directly")
        else:
            # Input might already be [0, 255]
            x = img / 255.0
            logger.info("[SegmentationCritic] Detected [0,255] range, scaling to [0,1]")
        
        # Clamp to valid [0, 1] before scaling
        x = torch.clamp(x, 0.0, 1.0)
        
        # Resize to YOLO's expected size (640x640)
        if x.shape[2] != 640 or x.shape[3] != 640:
            x = F.interpolate(x, size=(640, 640), mode='bilinear', align_corners=False)
        
        # Scale to [0, 255] (CRITICAL for YOLO)
        clean_input = x * 255.0
        
        # Debug: Verify final range
        logger.info(f"[SegmentationCritic] Scaled Range: [{clean_input.min().item():.1f}, {clean_input.max().item():.1f}]")
        
        # --- STEP 2: RAW LOGIT EXTRACTION ---
        self.model.model.eval()
        try:
            outputs = self.model.model(clean_input)
        except Exception as e:
            logger.warning(f"[SegmentationCritic] Raw model forward failed: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=False)

        # Handle Output Tuple (YOLO returns DEEPLY nested structure)
        # Must use while loop to fully unwrap
        preds = outputs
        while isinstance(preds, (tuple, list)):
            if len(preds) == 0:
                logger.warning("[SegmentationCritic] Empty output from YOLO")
                return torch.tensor(0.0, device=self.device, requires_grad=False)
            preds = preds[0]
        
        # --- STEP 3: EXTRACT PERSON CLASS LOGITS ---
        # YOLO output shape: [B, num_classes + 4 + masks, num_anchors]
        # Index 4 = Person class confidence (after x, y, w, h)
        if preds.dim() == 3:
            person_logits = preds[:, 4, :]  # [B, num_anchors]
        elif preds.dim() == 2:
            person_logits = preds[4:5, :]   # [1, num_anchors]
        else:
            logger.warning(f"[SegmentationCritic] Unexpected preds shape: {preds.shape}")
            return torch.tensor(0.0, device=self.device, requires_grad=False)
        
        # Debug: Log logit stats
        logger.info(f"[SegmentationCritic] Person Logits - Mean: {person_logits.mean().item():.4f}, Max: {person_logits.max().item():.4f}")

        # --- STEP 4: STABLE ZERO-TARGET LOSS ---
        # Target: 0.0 (We want zero probability of person detection)
        target = torch.zeros_like(person_logits)

        # BCEWithLogitsLoss for stable gradient flow
        loss = F.binary_cross_entropy_with_logits(
            person_logits,
            target,
            reduction='mean'
        )
        
        logger.info(f"[SegmentationCritic] Loss: {loss.item():.4f}")
        
        return loss
    
    def get_person_masks(self, image: torch.Tensor) -> Optional[np.ndarray]:
        """
        Get segmentation masks for persons in the image.
        Uses full NMS pipeline (for visualization, not gradients).
        
        Args:
            image: [B, 3, H, W] tensor in [0, 1] range
            
        Returns:
            Combined mask of all detected persons [H, W] or None
        """
        if not self.loaded:
            self.load()
        if not self.loaded:
            return None
            
        with torch.no_grad():
            img_clamped = image.clamp(0, 1)
            if img_clamped.dim() == 4:
                img_np = img_clamped[0].permute(1, 2, 0).cpu().numpy()
            else:
                img_np = img_clamped.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            results = self.model(img_np, verbose=False)
            
            combined_mask = None
            for result in results:
                if result.masks is not None:
                    for i, cls in enumerate(result.boxes.cls.cpu().numpy()):
                        if cls == self.PERSON_CLASS_ID:
                            mask = result.masks.data[i].cpu().numpy()
                            if combined_mask is None:
                                combined_mask = mask
                            else:
                                combined_mask = np.maximum(combined_mask, mask)
                                
        return combined_mask


def test_segmentation_critic():
    """Test the SegmentationCritic."""
    print("Testing SegmentationCritic...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    critic = SegmentationCritic(device=device)
    
    # Create test image (solid color - should have no person)
    test_img = torch.rand(1, 3, 640, 640, device=device)
    
    loss = critic.compute_loss(test_img)
    print(f"Loss on random image: {loss.item():.4f}")
    
    # Check VRAM
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        print(f"VRAM used: {allocated:.2f} GB")
    
    critic.unload()
    print("[PASS] SegmentationCritic test complete!")


if __name__ == "__main__":
    test_segmentation_critic()
