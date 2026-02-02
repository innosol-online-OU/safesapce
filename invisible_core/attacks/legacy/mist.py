
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from PIL import Image

class MistCloak:
    """
    Protocol D: "Mist" (Anti-Diffusion / Anti-GenAI)
    Target Environment: Generative AI (Stable Diffusion, Midjourney, etc.)
    Operational Goal: Disrupt Latent Representation (Prevent Editing/Inpainting)
    """

    def __init__(self, model_name='resnet50', device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[Mist] Initializing on {self.device}...")
        try:
            # We use a standard CNN (ResNet50) as a surrogate feature extractor.
            # Attacking ResNet features often transfers to the U-Net encoder of Diffusion models
            # because they share similar inductive biases for texture and shape.
            self.model = timm.create_model(model_name, pretrained=True).to(self.device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            self.model_loaded = True
        except Exception as e:
            print(f"[Mist] Failed to load model {model_name}: {e}")
            self.model_loaded = False

        self.preprocess = T.Compose([
            T.Resize((512, 512)), # Standard GenAI resolution
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.denormalize = T.Compose([
             T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
             T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
        ])

    def generate_mist(self, image: Image.Image, epsilon=0.03, iterations=40) -> Image.Image:
        """
        Generates "Mist" - adversarial noise targeting feature representations.
        Goal: Maximize distance in feature space within pixel constraints.
        """
        if not self.model_loaded:
            return image

        original_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        delta = torch.zeros_like(original_tensor, requires_grad=True).to(self.device)
        
        # PGD Optimizer
        optimizer = torch.optim.SGD([delta], lr=0.005)

        for i in range(iterations):
            optimizer.zero_grad()
            
            adv_input = original_tensor + delta
            
            # Extract features (using penultimate layer output usually, or just logits for simplicity here)
            # For better Mist, we should ideally tap into intermediate layers, but 'forward' gives logits.
            # Using logits distribution disruption is a decent proxy for "semantics disruption".
            outputs = self.model(adv_input)
            original_outputs = self.model(original_tensor)
            
            # Objective: Maximize distance from original representation (Feature Chaos)
            # We want the model to think this image is SOMETHING ELSE entirely.
            # MSE Loss between original and adversarial features -> We want to MAXIMIZE this.
            
            # We minimize negative distance
            loss = -1 * F.mse_loss(outputs, original_outputs)
            
            loss.backward()
            
            # Projected Gradient Descent
            grad = delta.grad.detach()
            delta.data = delta.data - 0.005 * grad.sign() # Move AWAY from original (Maximize error)
            
            # Clip
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.data = torch.clamp(original_tensor + delta.data, 0, 1) - original_tensor # Maintain stats (rough)

        # Reconstruct
        adv_tensor = (original_tensor + delta).detach().cpu().squeeze(0)
        adv_tensor = self.denormalize(adv_tensor)
        adv_tensor = torch.clamp(adv_tensor, 0, 1)
        
        adv_image = T.ToPILImage()(adv_tensor)
        adv_image = adv_image.resize(image.size)
        
        return adv_image
