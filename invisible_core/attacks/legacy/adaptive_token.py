
import torch
import torchvision.transforms as T
import timm
from PIL import Image

class AdaptiveTokenTuner:
    """
    Protocol A: "Digital-Ghost" (Adaptive Token Tuning)
    Target Environment: Digital / Web
    Operational Goal: Universal Evasion (Black-Box Transferability)
    """

    def __init__(self, model_name='deit_small_patch16_224', device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[Digital-Ghost] Initializing on {self.device}...")
        try:
            self.model = timm.create_model(model_name, pretrained=True).to(self.device)
            self.model.eval()
            # Disable gradients for model parameters
            for param in self.model.parameters():
                param.requires_grad = False
            self.model_loaded = True
        except Exception as e:
            print(f"[Digital-Ghost] Failed to load model {model_name}: {e}")
            self.model_loaded = False

        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            # ImageNet normalization
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.denormalize = T.Compose([
             T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
             T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
        ])

    def generate_perturbation(self, image: Image.Image, iterations=50, epsilon=8/255) -> Image.Image:
        """
        Generates adversarial perturbation using Adaptive Token Tuning.
        Strategies:
        1. Adaptive Gradient Re-scaling
        2. Self-Paced Patch Out
        3. Hybrid Token Gradient Truncation
        """
        if not self.model_loaded:
            print("[Digital-Ghost] Model not loaded, returning original image.")
            return image

        # Prepare input
        original_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        delta = torch.zeros_like(original_tensor, requires_grad=True).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.SGD([delta], lr=0.01)

        for i in range(iterations):
            optimizer.zero_grad()
            
            # Strategy 2: Self-Paced Patch Out (Randomly drop tokens/patches)
            # Simulating token drop by varying the effective input
            # In a real token-tuner, we'd hook into the embedding layer, 
            # here we can simulate by masking parts of the input image or noise.
            mask = torch.rand_like(delta) > 0.1 # Keep 90% of pixels/tokens
            
            adv_input = original_tensor + delta * mask.float()
            
            # Predict
            outputs = self.model(adv_input)
            
            # Maximize entropy / minimize confidence in top class (Untargeted Attack)
            # Simply trying to maximize the distance from original prediction or Minimize max logit
            loss = -1 * torch.max(outputs, dim=1)[0].mean() # Minimize confidence of top class
            
            loss.backward()

            # Strategy 1 & 3: Gradient manipulation
            grad = delta.grad.clone()
            
            # Strategy 3: Hybrid Token Gradient Truncation
            # Truncate gradients > 95th percentile
            threshold = torch.quantile(torch.abs(grad), 0.95)
            grad[torch.abs(grad) > threshold] = 0 # Zero out extreme gradients
            
            # Strategy 1: Adaptive Gradient Re-scaling
            # Normalize gradients based on variance (simplified global normalization here)
            grad_std = torch.std(grad) + 1e-9
            grad = grad / grad_std

            # Update delta manually to apply our modified gradient
            delta.data = delta.data - 0.01 * grad.sign() # FGSM-like step with modified grad
            
            # Clip delta
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.data = torch.clamp(original_tensor + delta.data, 0, 1) - original_tensor # Maintain pixel range

        # Reconstruct image
        adv_tensor = (original_tensor + delta).detach().cpu().squeeze(0)
        # Denormalize to get back to 0-1 range approx (ignoring norm effect for exact pixel regen)
        # For visualization/saving, we need to reverse the ImageNet normalization if we want standard RGB
        # But wait, original_tensor was normalized. So we must denormalize.
        adv_tensor = self.denormalize(adv_tensor)
        adv_tensor = torch.clamp(adv_tensor, 0, 1)
        
        adv_image = T.ToPILImage()(adv_tensor)
        
        # Resize back to original size
        adv_image = adv_image.resize(image.size)
        
        return adv_image

