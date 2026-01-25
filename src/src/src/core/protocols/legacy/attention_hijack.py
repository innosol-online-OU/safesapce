
import torch
import torch.nn as nn
import timm
from PIL import Image
import numpy as np
import torchvision.transforms as T

class AttentionHijacker:
    """
    Protocol C: "Attention-Hijack" (Patch-Fool)
    Target Environment: High-Security / White-Box Systems
    Operational Goal: Model Breakage (Architecture Exploitation)
    """

    def __init__(self, model_name='deit_small_patch16_224', device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[Attention-Hijack] Initializing on {self.device}...")
        try:
            self.model = timm.create_model(model_name, pretrained=True).to(self.device)
            self.model.eval()
             # Disable gradients for model parameters
            for param in self.model.parameters():
                param.requires_grad = False
            self.model_loaded = True
        except Exception as e:
            print(f"[Attention-Hijack] Failed to load model {model_name}: {e}")
            self.model_loaded = False
            
        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.denormalize = T.Compose([
             T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
             T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
        ])

    def optimize_patch(self, image: Image.Image, mode='dense', iterations=50) -> Image.Image:
        """
        Optimizes a patch (or sparse pixels) to hijack attention.
        Objective: Force Class Token (CLS) to attend to the patch.
        Modes: 'dense' (contiguous), 'sparse' (scattered dust).
        """
        if not self.model_loaded:
            return image

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Initialize patch/noise
        if mode == 'dense':
            # Contiguous small patch
            # Mask handles the location
            mask = torch.zeros_like(image_tensor)
            h, w = mask.shape[2:]
            patch_h, patch_w = 32, 32
            # Center patch
            mask[:, :, h//2 - patch_h//2 : h//2 + patch_h//2, w//2 - patch_w//2 : w//2 + patch_w//2] = 1.0
        else: # sparse
            # 250 random pixels
            mask = torch.zeros_like(image_tensor)
            indices = torch.randperm(mask.numel())[:250 * 3] # *3 for RGB channels roughly
            mask.view(-1)[indices] = 1.0

        delta = torch.zeros_like(image_tensor, requires_grad=True).to(self.device)
        optimizer = torch.optim.SGD([delta], lr=0.1)
        
        # Hook to capture attention weights
        # Disclaimer: actual hooking depends heavily on specific timm model structure.
        # This is a conceptual implementation assuming we can access attention.
        attention_scores = []
        def attn_hook(module, input, output):
            # output might be (B, N, N) attention matrix or tuple
            # We assume standard attention map (Batch, Heads, Tokens, Tokens)
            # We want attention of CLS token (index 0) to other tokens.
            # Usually output is just the attention weights if we hook the right place 
            # or we need to re-compute it if we hook the linear layers.
            # Simplified: Assuming we can get 'attn' from somewhere.
            pass

        # Since we can't easily hook without knowing the exact model internals structure guaranteed to match 'deit_small_patch16_224'
        # in this blind environment, we will define a proxy loss function:
        # "Maximize activation of the patched area features" as a proxy for attention hijacking.
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            adv_img = image_tensor * (1 - mask) + (image_tensor + delta) * mask
            
            outputs = self.model(adv_img)
            
            # Objective 1: Attention Maximization (The "Black Hole")
            # Ideally: loss = -sum(Attention(CLS, Patch_Tokens))
            # Proxy: Minimize confidence in original class or Maximize entropy strongly, 
            # PLUS minimize the norms of features outside the patch (suppression).
            
            # We will use a "Break output" loss - force output to be uniform distribution
            probs = torch.softmax(outputs, dim=1)
            loss = -1 * (probs * torch.log(probs + 1e-9)).sum() # Maximize entropy (make model confused)
            
            # Add term to maximize activation magnitude in the patch area (likely where attention will go)
            loss += -1 * torch.mean(torch.abs(adv_img * mask)) 
            
            loss.backward()
            optimizer.step()
            
            # Clip
            delta.data.clamp_(-1, 1)

        # Reconstruct
        final_tensor = image_tensor * (1 - mask) + (image_tensor + delta) * mask
        final_tensor = self.denormalize(final_tensor.detach().cpu().squeeze(0))
        final_tensor = torch.clamp(final_tensor, 0, 1)
        
        return T.ToPILImage()(final_tensor)
