"""
Project Invisible Phase 8: CLIP Semantic Critic (Anti-Identity)
Uses CLIP ViT-B/32 to minimize similarity between image and target identity text.
Attacks the CONCEPT of identity, not just shape/texture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from src.core.logger import logger


class CLIPCritic:
    """
    Semantic Identity Critic using CLIP.
    
    Goal: Minimize similarity between image and target text (e.g., "Elon Musk").
    This breaks AI's ability to recognize the specific identity in the image.
    """
    
    # CLIP normalization constants
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
    
    def __init__(self, device: str = 'cuda', target_text: str = "a person"):
        self.device = device
        self.model = None
        self.processor = None
        self.loaded = False
        self.target_text = target_text
        self.target_embedding = None
        
        # Pre-compute normalization tensors
        self.mu = torch.tensor(self.CLIP_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(self.CLIP_STD).view(1, 3, 1, 1)
        
    def load(self):
        """Lazy load CLIP ViT-B/32 model."""
        if self.loaded:
            return
            
        logger.info("[CLIPCritic] Loading CLIP ViT-L/14@336px (High Fidelity)...")
        try:
            from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
            
            # Phase 13.5 Upgrade: Use Large Model @ 336px for detailed texture guidance
            model_id = "openai/clip-vit-large-patch14-336"
            
            self.model = CLIPModel.from_pretrained(
                model_id, 
                use_safetensors=True,
                token="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"
            ).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(
                model_id, 
                use_safetensors=True, 
                use_fast=True,
                token="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"
            )
            self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
            
            # Move normalization tensors to device
            self.mu = self.mu.to(self.device)
            self.std = self.std.to(self.device)
            
            self.loaded = True
            logger.info("[CLIPCritic] Model loaded successfully.")
            
            # Pre-compute target embedding
            self._update_target_embedding()
            
        except Exception as e:
            logger.error(f"[CLIPCritic] Failed to load: {e}")
            self.loaded = False
            
    def _update_target_embedding(self):
        """Compute and cache the target text embedding."""
        if not self.loaded or self.model is None:
            return
            
        logger.info(f"[CLIPCritic] Computing embedding for target: '{self.target_text}'")
        
        with torch.no_grad():
            # Tokenize target text
            inputs = self.tokenizer(
                self.target_text, 
                return_tensors="pt", 
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Get text embedding
            text_features = self.model.get_text_features(**inputs)
            self.target_embedding = text_features / text_features.norm(dim=-1, keepdim=True)
            
        logger.info(f"[CLIPCritic] Target embedding cached (shape: {self.target_embedding.shape})")
    
    def set_target(self, target_text: str):
        """Update the target identity text."""
        if target_text != self.target_text:
            self.target_text = target_text
            if self.loaded:
                self._update_target_embedding()
            
    def unload(self):
        """Free VRAM."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.target_embedding = None
            self.loaded = False
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[CLIPCritic] Model unloaded.")
    
    def compute_loss(self, image: torch.Tensor, target_text: Optional[str] = None, 
                      focus_box: Optional[Tuple[int, int, int, int]] = None) -> torch.Tensor:
        """
        Compute semantic similarity loss between image and target text.
        
        Phase 8.2: SURGICAL STRIKE
        When focus_box is provided, crops to that region before computing similarity.
        This focuses gradient energy on identity-critical pixels (face).
        
        Args:
            image: [B, 3, H, W] tensor (can be [-1, 1] or [0, 1] range)
            target_text: Optional override for target identity
            focus_box: Optional (y1, x1, y2, x2) bounding box to crop before analysis
            
        Returns:
            loss: Cosine similarity (minimize this to erase identity)
        """
        if not self.loaded:
            self.load()
        if not self.loaded:
            return torch.tensor(0.0, device=self.device, requires_grad=False)
        
        # Update target if changed
        if target_text and target_text != self.target_text:
            self.set_target(target_text)
        
        # Ensure image is on correct device
        img = image.to(self.device)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        
        # --- PHASE 8.2: SURGICAL CROP ---
        # If focus_box provided, crop to face/person region
        if focus_box is not None:
            y1, x1, y2, x2 = focus_box
            # Ensure valid bounds
            y1 = max(0, min(y1, img.shape[2] - 1))
            y2 = max(y1 + 1, min(y2, img.shape[2]))
            x1 = max(0, min(x1, img.shape[3] - 1))
            x2 = max(x1 + 1, min(x2, img.shape[3]))
            
            # Crop to face region (differentiable)
            img = img[:, :, y1:y2, x1:x2]
            logger.info(f"[CLIPCritic] Surgical crop: ({y1},{x1}) to ({y2},{x2})")
        
        # --- STEP 1: NORMALIZE INPUT ---
        # Auto-detect and normalize to [0, 1]
        img_min = img.min().item()
        
        if img_min < -0.5:
            # Input is [-1, 1] range
            x = (img + 1.0) / 2.0
        else:
            # Input is [0, 1] range
            x = img
        
        x = torch.clamp(x, 0.0, 1.0)
        
        # --- STEP 2: RESIZE TO CLIP SIZE (336x336 for ViT-L/14@336px) ---
        if x.shape[2] != 336 or x.shape[3] != 336:
            x = F.interpolate(x, size=(336, 336), mode='bilinear', align_corners=False)
        
        # --- STEP 3: APPLY CLIP NORMALIZATION ---
        # CLIP expects specific mean/std normalization
        clip_input = (x - self.mu) / self.std
        
        # --- STEP 4: GET IMAGE FEATURES ---
        image_features = self.model.get_image_features(pixel_values=clip_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # --- STEP 5: COMPUTE COSINE SIMILARITY ---
        # We MINIMIZE this to make image look LESS like target
        similarity = F.cosine_similarity(image_features, self.target_embedding)
        
        # Debug logging (less verbose)
        if focus_box:
            logger.info(f"[CLIPCritic] Face Similarity to '{self.target_text}': {similarity.mean().item():.4f}")
        else:
            logger.info(f"[CLIPCritic] Full Image Similarity: {similarity.mean().item():.4f}")
        
        # Return similarity as loss (minimize to erase identity)
        return similarity.mean()
    
    def get_similarity(self, image: torch.Tensor) -> float:
        """Get similarity score without gradients (for validation)."""
        with torch.no_grad():
            return self.compute_loss(image).item()


def test_clip_critic():
    """Test the CLIPCritic."""
    print("Testing CLIPCritic...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    critic = CLIPCritic(device=device, target_text="Elon Musk")
    
    # Create test image (random - should have low similarity to any person)
    test_img = torch.rand(1, 3, 224, 224, device=device)
    
    similarity = critic.compute_loss(test_img)
    print(f"Similarity to 'Elon Musk' on random image: {similarity.item():.4f}")
    
    # Check VRAM
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        print(f"VRAM used: {allocated:.2f} GB")
    
    critic.unload()
    print("[PASS] CLIPCritic test complete!")


if __name__ == "__main__":
    test_clip_critic()
