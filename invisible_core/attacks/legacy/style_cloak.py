
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from PIL import Image

class StyleCloak:
    """
    Protocol: "Style Shield" (Anti-Mimicry / Glaze-like)
    Target: Style Transfer Models / LoRA Adapters
    Goal: Shift style features while preserving content.
    """

    def __init__(self, model_name='vgg19', device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[StyleCloak] Initializing on {self.device}...")
        try:
            # VGG19 Features
            self.model = timm.create_model(model_name, pretrained=True, features_only=True).to(self.device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            self.model_loaded = True
        except Exception as e:
            print(f"[StyleCloak] Failed to load model {model_name}: {e}")
            self.model_loaded = False

        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Denormalize is CRITICAL for saving visualization correctly
        # Revert ImageNet Stats (Fixes Grey Image)
        # pixel = (tensor * std) + mean
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

    def gram_matrix(self, input):
        a, b, c, d = input.size() 
        features = input.view(a * b, c * d) 
        G = torch.mm(features, features.t()) 
        return G.div(a * b * c * d)

    def generate_cloak(self, image: Image.Image, iterations=40, intensity=0.1) -> Image.Image:
        """
        Shifts styles towards random target.
        """
        if not self.model_loaded:
            return image

        original_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        delta = torch.zeros_like(original_tensor, requires_grad=True).to(self.device)
        
        optimizer = torch.optim.SGD([delta], lr=0.05)

        # "Alien Style" Target (Random Noise features)
        alien_style_input = torch.randn_like(original_tensor).to(self.device)
        alien_features = self.model(alien_style_input)
        if not isinstance(alien_features, list):
            alien_features = [alien_features]
        target_grams = [self.gram_matrix(f).detach() for f in alien_features]

        for i in range(iterations):
            optimizer.zero_grad()
            adv_input = original_tensor + delta
            
            features = self.model(adv_input)
            if not isinstance(features, list):
                features = [features]
            
            style_loss = 0
            for ft, tg in zip(features, target_grams):
                gm = self.gram_matrix(ft)
                style_loss += F.mse_loss(gm, tg)
            
            style_loss.backward()
            optimizer.step()
            
            # Clip perturbation
            delta.data.clamp_(-intensity, intensity)
            
        # Reconstruct & Denormalize CRITICAL STEP
        adv_tensor = (original_tensor + delta).detach().clone().squeeze(0)
        
        # Manual Denormalization Logic
        # Result = (Normed * Std) + Mean
        adv_tensor = adv_tensor * self.std + self.mean
        adv_tensor = torch.clamp(adv_tensor, 0, 1)
        
        # Convert to PIL
        adv_image = T.ToPILImage()(adv_tensor.cpu())
        adv_image = adv_image.resize(image.size)
        
        return adv_image
