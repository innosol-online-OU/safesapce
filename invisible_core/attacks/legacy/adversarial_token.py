
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class AdversarialTokenInjector:
    """
    Protocol: "Adversarial Token Injection" (Invisible Compliance)
    Target: CLIP / Vision-Language Models
    Goal: Force image embedding to align with "GDPR_BLOCK" token strictly via adversarial perturbation.
    Constraint: Invisible (L_infinity < 4/255) + JPEG Robustness
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32", device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[AdversarialToken] Initializing on {self.device}...")
        
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            self.model_loaded = True
        except Exception as e:
            print(f"[AdversarialToken] Failed to load CLIP: {e}")
            self.model_loaded = False

        # Stats for normalization (CLIP usually expects 0-1 or specific mean/std depending on processor)
        # CLIPProcessor usually handles normalization internally if we give it PIL, 
        # but for PGD we need manual tensor control.
        # OpenAI CLIP mean/std
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(self.device)

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def apply(self, image: Image.Image, target_text="SYSTEM ALERT: BANNED CONTENT", eps=4/255, alpha=1/255, steps=50) -> Image.Image:
        if not self.model_loaded:
            return image

        # 1. Prepare Full Resolution Tensor
        original_tensor = T.ToTensor()(image).to(self.device).unsqueeze(0) # (1, 3, H, W)
        
        # Delta matches original resolution
        delta = torch.zeros_like(original_tensor, requires_grad=True).to(self.device)
        optimizer = torch.optim.SGD([delta], lr=alpha)
        
        target_emb = self.get_text_embedding(target_text)
        
        print(f"[AdversarialToken] Optimizing for token: '{target_text}' (Full Res)")

        for i in range(steps):
            optimizer.zero_grad()
            
            # Apply Delta
            adv_full = original_tensor + delta
            
            # 2. Resize to 224x224 for CLIP Loss
            # Interpolation Smoothing is naturally handled here if we jitter the size, 
            # but standard bilinear downsampling is enough for basic robustness.
            adv_resized = F.interpolate(adv_full, size=(224, 224), mode='bilinear', align_corners=False)
            
            # Normalization for CLIP
            norm_adv = (adv_resized.squeeze(0) - self.mean) / self.std
            
            # Random Interpolation Smoothing (JPEG Robustness) technique
            if i % 3 == 0:
                 # Jitter the resized version slightly before loss
                 scale = 0.9 + (0.2 * torch.rand(1).item())
                 h, w = 224, 224
                 sh, sw = int(h*scale), int(w*scale)
                 norm_adv = F.interpolate(norm_adv.unsqueeze(0), size=(sh, sw), mode='bilinear', align_corners=False)
                 norm_adv = F.interpolate(norm_adv, size=(h, w), mode='bilinear', align_corners=False).squeeze(0)

            # Get Image Embedding
            image_features = self.model.get_image_features(pixel_values=norm_adv.unsqueeze(0))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Loss: 1 - CosineSimilarity
            loss = 1 - torch.cosine_similarity(image_features, target_emb)
            
            loss.backward()
            optimizer.step()
            
            # Projection (L_inf constraint) on Full Res Delta
            delta.data.clamp_(-eps, eps)
            delta.data = torch.clamp(original_tensor + delta.data, 0, 1) - original_tensor
            
        # Final Output (Full Res)
        final_tensor = (original_tensor + delta).detach().cpu().squeeze(0)
        return T.ToPILImage()(final_tensor)
