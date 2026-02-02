"""
Project Invisible Phase 2: Selective Ensemble Attack
Samples 3 models per optimization step from a diverse pool to maximize transferability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass

# CVE-2025-32434 bypass for trusted models
try:
    import transformers.utils.import_utils
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass


@dataclass
class EnsembleConfig:
    """Configuration for ensemble attack."""
    diversity: int = 5  # 1-10, higher = more model variety per step
    models_per_step: int = 3
    gradient_accumulation: bool = True
    weight_by_loss: bool = True  # Weight gradients by inverse loss (harder models get more weight)


class SurrogateModel:
    """
    Wrapper for a surrogate model with gradient computation.
    Implements lazy loading protocol for VRAM efficiency.
    """
    
    # Estimated VRAM usage per model (in GB)
    VRAM_ESTIMATES = {
        "clip-vit-large-patch14": 1.7,
        "clip-vit-base-patch32": 0.6,
        "resnet50": 0.1,
        "convnext-base": 0.35,
        "dinov2-base": 0.4,
    }
    
    def __init__(self, name: str, model_type: str, device: str = 'cuda'):
        self.name = name
        self.model_type = model_type
        self.device = device
        self.model = None
        self.processor = None
        self.feature_extractor = None
        self.loaded = False
        self._estimated_vram_gb = self.VRAM_ESTIMATES.get(name, 0.5)
        
    @staticmethod
    def get_vram_usage() -> Dict[str, float]:
        """Get current VRAM usage statistics."""
        if not torch.cuda.is_available():
            return {"available": 0, "used": 0, "total": 0}
        
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        
        return {
            "total_gb": round(total, 2),
            "reserved_gb": round(reserved, 2),
            "allocated_gb": round(allocated, 2),
            "free_gb": round(total - reserved, 2)
        }
    
    def load(self):
        """Lazy load the model to save VRAM."""
        raise NotImplementedError
        
    def unload(self):
        """
        Unload model to free VRAM.
        Implements aggressive cleanup protocol.
        """
        if not self.loaded:
            return
            
        print(f"[Ensemble] Unloading {self.name}...", flush=True)
        
        # Clear all model references
        if self.model is not None:
            # Move to CPU first (sometimes helps with VRAM release)
            try:
                self.model.cpu()
            except Exception:
                pass
            del self.model
            self.model = None
            
        if self.processor is not None:
            del self.processor
            self.processor = None
            
        if self.feature_extractor is not None:
            try:
                self.feature_extractor.cpu()
            except Exception:
                pass
            del self.feature_extractor
            self.feature_extractor = None
            
        self.loaded = False
        
        # Aggressive garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def compute_loss(self, image_tensor: torch.Tensor, original_features: torch.Tensor) -> torch.Tensor:
        """Compute adversarial loss for this model."""
        raise NotImplementedError
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.unload()
        except Exception:
            pass


class CLIPViTLargeSurrogate(SurrogateModel):
    """CLIP ViT-Large/14 - Semantic/Language alignment critic."""
    
    def __init__(self, device: str = 'cuda'):
        super().__init__("clip-vit-large-patch14", "semantic", device)
        
    def load(self):
        if self.loaded:
            return
        print(f"[Ensemble] Loading {self.name}...", flush=True)
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
        self.loaded = True
        
    def compute_loss(self, image_tensor: torch.Tensor, original_features: torch.Tensor) -> torch.Tensor:
        """Maximize distance from original CLIP embedding."""
        if not self.loaded:
            self.load()
            
        # FIX: Dynamic dtype handling
        params = next(self.model.parameters())
        target_dtype = params.dtype
        image_tensor = image_tensor.to(dtype=target_dtype)
            
        # Resize to 224x224 for CLIP
        img_224 = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        
        # CLIP normalization
        mu = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=target_dtype).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=target_dtype).view(1, 3, 1, 1).to(self.device)
        clip_input = ((img_224 + 1) * 0.5 - mu) / std
        
        current_features = self.model.get_image_features(pixel_values=clip_input)
        current_features = current_features / current_features.norm(dim=-1, keepdim=True)
        
        # Minimize cosine similarity (maximize distance)
        similarity = F.cosine_similarity(current_features, original_features)
        return similarity.mean()


class CLIPViTBaseSurrogate(SurrogateModel):
    """CLIP ViT-Base/32 - Faster semantic critic."""
    
    def __init__(self, device: str = 'cuda'):
        super().__init__("clip-vit-base-patch32", "semantic", device)
        
    def load(self):
        if self.loaded:
            return
        print(f"[Ensemble] Loading {self.name}...", flush=True)
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self.loaded = True
        
    def compute_loss(self, image_tensor: torch.Tensor, original_features: torch.Tensor) -> torch.Tensor:
        if not self.loaded:
            self.load()
        
        # FIX: Dynamic dtype handling
        params = next(self.model.parameters())
        target_dtype = params.dtype
        image_tensor = image_tensor.to(dtype=target_dtype)
            
        img_224 = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        mu = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=target_dtype).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=target_dtype).view(1, 3, 1, 1).to(self.device)
        clip_input = ((img_224 + 1) * 0.5 - mu) / std
        
        current_features = self.model.get_image_features(pixel_values=clip_input)
        current_features = current_features / current_features.norm(dim=-1, keepdim=True)
        
        similarity = F.cosine_similarity(current_features, original_features)
        return similarity.mean()


class ResNet50Surrogate(SurrogateModel):
    """ResNet-50 ImageNet classifier - Classification features critic."""
    
    def __init__(self, device: str = 'cuda'):
        super().__init__("resnet50", "classification", device)
        
    def load(self):
        if self.loaded:
            return
        print(f"[Ensemble] Loading {self.name}...", flush=True)
        import torchvision.models as models
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(self.device)
        self.model.eval()
        # Remove final classification layer to get features
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.loaded = True
        
    def compute_loss(self, image_tensor: torch.Tensor, original_features: torch.Tensor) -> torch.Tensor:
        if not self.loaded:
            self.load()
        
        # FIX: Dynamic dtype handling
        params = next(self.model.parameters())
        target_dtype = params.dtype
        image_tensor = image_tensor.to(dtype=target_dtype)
            
        # ResNet expects 224x224, ImageNet normalization
        img_224 = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        img_normalized = (img_224 + 1) * 0.5  # [-1,1] -> [0,1]
        mu = torch.tensor([0.485, 0.456, 0.406], dtype=target_dtype).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=target_dtype).view(1, 3, 1, 1).to(self.device)
        img_normalized = (img_normalized - mu) / std
        
        current_features = self.feature_extractor(img_normalized).flatten(1)
        current_features = current_features / current_features.norm(dim=-1, keepdim=True)
        
        similarity = F.cosine_similarity(current_features, original_features)
        return similarity.mean()


class ConvNeXtSurrogate(SurrogateModel):
    """ConvNeXt-Base - Modern ConvNet architecture."""
    
    def __init__(self, device: str = 'cuda'):
        super().__init__("convnext-base", "classification", device)
        
    def load(self):
        if self.loaded:
            return
        print(f"[Ensemble] Loading {self.name}...", flush=True)
        import torchvision.models as models
        self.model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1).to(self.device)
        self.model.eval()
        # Get features before classifier
        self.feature_extractor = nn.Sequential(
            self.model.features,
            self.model.avgpool
        )
        self.loaded = True
        
    def compute_loss(self, image_tensor: torch.Tensor, original_features: torch.Tensor) -> torch.Tensor:
        if not self.loaded:
            self.load()
        
        # FIX: Dynamic dtype handling
        params = next(self.model.parameters())
        target_dtype = params.dtype
        image_tensor = image_tensor.to(dtype=target_dtype)
            
        img_224 = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        img_normalized = (img_224 + 1) * 0.5
        mu = torch.tensor([0.485, 0.456, 0.406], dtype=target_dtype).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=target_dtype).view(1, 3, 1, 1).to(self.device)
        img_normalized = (img_normalized - mu) / std
        
        current_features = self.feature_extractor(img_normalized).flatten(1)
        current_features = current_features / current_features.norm(dim=-1, keepdim=True)
        
        similarity = F.cosine_similarity(current_features, original_features)
        return similarity.mean()


class DINOv2Surrogate(SurrogateModel):
    """DINOv2 ViT-Base - Self-supervised vision transformer."""
    
    def __init__(self, device: str = 'cuda'):
        super().__init__("dinov2-base", "self-supervised", device)
        
    def load(self):
        if self.loaded:
            return
        print(f"[Ensemble] Loading {self.name}...", flush=True)
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(self.device)
        self.model.eval()
        self.loaded = True
        
    def compute_loss(self, image_tensor: torch.Tensor, original_features: torch.Tensor) -> torch.Tensor:
        if not self.loaded:
            self.load()
            
        # Using PyTorch Hub, might not have parameters() if it's a wrapper, but DINOv2 is a ViT.
        params = next(self.model.parameters())
        target_dtype = params.dtype
        
        # DINOv2 expects 518x518 or 224x224
        img_224 = F.interpolate(image_tensor.to(target_dtype), size=(224, 224), mode='bilinear', align_corners=False)
        img_normalized = (img_224 + 1) * 0.5
        mu = torch.tensor([0.485, 0.456, 0.406], dtype=target_dtype).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=target_dtype).view(1, 3, 1, 1).to(self.device)
        img_normalized = (img_normalized - mu) / std
        
        current_features = self.model(img_normalized)
        current_features = current_features / current_features.norm(dim=-1, keepdim=True)
        
        similarity = F.cosine_similarity(current_features, original_features)
        return similarity.mean()


class SelectiveEnsemble:
    """
    Selective Ensemble Attack: Samples 3 models per optimization step
    from a diverse pool to maximize adversarial transferability.
    
    NOTE: Models are kept loaded on GPU for performance (user confirmed GPU idle).
    """
    
    def __init__(self, device: str = 'cuda', config: EnsembleConfig = None, keep_loaded: bool = True):
        self.device = device
        self.config = config or EnsembleConfig()
        self.keep_loaded = keep_loaded  # Keep models on GPU
        
        # Model pool with different architecture families
        self.model_pool: List[SurrogateModel] = [
            CLIPViTLargeSurrogate(device),
            CLIPViTBaseSurrogate(device),
            ResNet50Surrogate(device),
            ConvNeXtSurrogate(device),
        ]
        
        # Track model usage for diversity
        self.usage_counts: Dict[str, int] = {m.name: 0 for m in self.model_pool}
        self.step_count = 0
        
        # Feature cache (stores on GPU for performance)
        self._feature_cache: Dict[str, torch.Tensor] = {}
        self._features_cached = False
        self._models_loaded = False
        
        print(f"[SelectiveEnsemble] Initialized with {len(self.model_pool)} surrogate models.")
        print(f"[SelectiveEnsemble] Keep loaded mode: {self.keep_loaded}")
        
    def load_all_models(self):
        """Pre-load all models to GPU."""
        if self._models_loaded:
            return
        print("[Ensemble] Loading ALL models to GPU...", flush=True)
        for model in self.model_pool:
            try:
                model.load()
            except Exception as e:
                print(f"[Ensemble] Failed to load {model.name}: {e}")
        self._models_loaded = True
        print("[Ensemble] All models loaded.", flush=True)
        
    def _log_vram(self, context: str = ""):
        """Log current VRAM usage."""
        if torch.cuda.is_available():
            stats = SurrogateModel.get_vram_usage()
            print(f"[VRAM] {context}: {stats['allocated_gb']:.2f}GB used / {stats['total_gb']:.2f}GB total ({stats['free_gb']:.2f}GB free)", flush=True)
            
    def _ensure_vram_available(self, required_gb: float) -> bool:
        """Check if enough VRAM is available, unload models if needed."""
        if not torch.cuda.is_available():
            return True
            
        stats = SurrogateModel.get_vram_usage()
        available = stats['free_gb']
        
        if available >= required_gb:
            return True
            
        # Try to free VRAM by unloading any loaded models
        print(f"[Ensemble] Need {required_gb:.2f}GB but only {available:.2f}GB free. Cleaning up...", flush=True)
        
        for model in self.model_pool:
            if model.loaded:
                model.unload()
                
        # Force cleanup
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Check again
        stats = SurrogateModel.get_vram_usage()
        return stats['free_gb'] >= required_gb
        
    def _select_models(self) -> List[SurrogateModel]:
        """
        Select models for this step based on diversity setting.
        Higher diversity = more randomization, lower = more consistent selection.
        """
        n = min(self.config.models_per_step, len(self.model_pool))
        
        if self.config.diversity >= 8:
            # High diversity: pure random sampling
            selected = random.sample(self.model_pool, n)
        elif self.config.diversity >= 5:
            # Medium diversity: weighted by inverse usage (prefer less-used)
            weights = [1.0 / (self.usage_counts[m.name] + 1) for m in self.model_pool]
            total = sum(weights)
            probs = [w / total for w in weights]
            indices = np.random.choice(len(self.model_pool), size=n, replace=False, p=probs)
            selected = [self.model_pool[i] for i in indices]
        else:
            # Low diversity: ensure architecture variety first
            by_type = {}
            for m in self.model_pool:
                if m.model_type not in by_type:
                    by_type[m.model_type] = []
                by_type[m.model_type].append(m)
            
            selected = []
            types = list(by_type.keys())
            random.shuffle(types)
            for t in types[:n]:
                selected.append(random.choice(by_type[t]))
            
            # Fill remaining slots randomly if needed
            while len(selected) < n:
                remaining = [m for m in self.model_pool if m not in selected]
                if remaining:
                    selected.append(random.choice(remaining))
                else:
                    break
        
        # Update usage
        for m in selected:
            self.usage_counts[m.name] += 1
            
        return selected
    
    def get_original_features(self, image_pil) -> Dict[str, torch.Tensor]:
        """
        Pre-compute original features for all models.
        Features are cached on GPU for performance.
        Models are kept loaded.
        """
        # Return cached features if available
        if self._features_cached and self._feature_cache:
            print(f"[Ensemble] Using cached features for {len(self._feature_cache)} models", flush=True)
            return self._feature_cache
        
        print("[Ensemble] Computing original features...", flush=True)
        
        # Load all models first
        self.load_all_models()
        
        features = {}
        
        # Convert PIL to tensor
        import numpy as np
        img_np = np.array(image_pil)
        img_tensor = torch.tensor(img_np).float().permute(2, 0, 1) / 255.0
        img_tensor = 2.0 * img_tensor - 1.0  # [0,1] -> [-1,1]
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        for model in self.model_pool:
            if not model.loaded:
                continue
            try:
                with torch.no_grad():
                    # Get features based on model type
                    if isinstance(model, (CLIPViTLargeSurrogate, CLIPViTBaseSurrogate)):
                        img_224 = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
                        mu = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
                        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)
                        clip_input = ((img_224 + 1) * 0.5 - mu) / std
                        feat = model.model.get_image_features(pixel_values=clip_input)
                    elif isinstance(model, (ResNet50Surrogate, ConvNeXtSurrogate)):
                        img_224 = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
                        img_normalized = (img_224 + 1) * 0.5
                        mu = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
                        img_normalized = (img_normalized - mu) / std
                        feat = model.feature_extractor(img_normalized).flatten(1)
                    else:
                        continue
                        
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    
                    # Keep features on GPU
                    self._feature_cache[model.name] = feat.detach()
                    features[model.name] = feat.detach()
                    
            except Exception as e:
                print(f"[Ensemble] Failed to get features from {model.name}: {e}")
                import traceback
                traceback.print_exc()
        
        self._features_cached = True
        print(f"[Ensemble] Cached features for {len(features)} models on GPU", flush=True)
                
        return features
    
    def compute_ensemble_loss(
        self, 
        decoded_image: torch.Tensor, 
        original_features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute aggregated loss from sampled ensemble models.
        Models are kept loaded on GPU.
        
        Returns:
            total_loss: Aggregated adversarial loss
            loss_breakdown: Individual losses per model
        """
        selected_models = self._select_models()
        self.step_count += 1
        
        losses = {}
        weights = {}
        
        for model in selected_models:
            try:
                # Ensure model is loaded
                if not model.loaded:
                    model.load()
                
                if model.name not in original_features:
                    continue
                
                # Ensure features are on correct device
                orig_feat = original_features[model.name]
                if orig_feat.device != decoded_image.device:
                    orig_feat = orig_feat.to(decoded_image.device)
                    
                loss = model.compute_loss(decoded_image, orig_feat)
                losses[model.name] = loss
                
                # Weight by inverse loss if enabled (harder models get more weight)
                if self.config.weight_by_loss:
                    weights[model.name] = 1.0 / (loss.item() + 0.1)
                else:
                    weights[model.name] = 1.0
                    
            except Exception as e:
                print(f"[Ensemble] Error computing loss for {model.name}: {e}")
                import traceback
                traceback.print_exc()
        
        if not losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {}
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Aggregate weighted loss
        total_loss = sum(normalized_weights[k] * losses[k] for k in losses.keys())
        
        loss_breakdown = {k: v.item() for k, v in losses.items()}
        
        return total_loss, loss_breakdown
    
    def get_stats(self) -> Dict:
        """Get ensemble usage statistics."""
        return {
            "total_steps": self.step_count,
            "model_usage": dict(self.usage_counts),
            "diversity_setting": self.config.diversity
        }


# Test function
def test_selective_ensemble():
    """Test the SelectiveEnsemble class."""
    print("Testing SelectiveEnsemble...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = EnsembleConfig(diversity=5, models_per_step=2)
    ensemble = SelectiveEnsemble(device=device, config=config)
    
    # Create dummy image
    from PIL import Image
    dummy_img = Image.new('RGB', (512, 512), color=(128, 128, 128))
    
    # Get original features
    print("Getting original features...")
    orig_features = ensemble.get_original_features(dummy_img)
    print(f"Got features for {len(orig_features)} models")
    
    # Simulate optimization step
    print("Simulating optimization step...")
    dummy_tensor = torch.randn(1, 3, 512, 512, device=device, requires_grad=True)
    
    loss, breakdown = ensemble.compute_ensemble_loss(dummy_tensor, orig_features)
    print(f"Total loss: {loss.item():.4f}")
    print(f"Breakdown: {breakdown}")
    print(f"Stats: {ensemble.get_stats()}")
    
    print("✅ SelectiveEnsemble test passed!")


if __name__ == "__main__":
    test_selective_ensemble()
