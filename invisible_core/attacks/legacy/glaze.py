
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from PIL import Image

class GlazeCloak:
    """
    Protocol E: "Glaze" (Style Disruption)
    Target Environment: Style Transfer / LoRA Adpaters
    Operational Goal: Style Mimicry Prevention (Shift Style Representation)
    """

    def __init__(self, model_name='vgg19', device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[Glaze] Initializing on {self.device}...")
        try:
            # VGG19 is standard for Style Loss
            # We use timm's vgg19 or torchvision's 
            self.model = timm.create_model(model_name, pretrained=True).to(self.device)
            # We need intermediate layers for Gram Matrix calculation
            # timm models return features. Ideally we want multi-scale features.
            # For simplicity in this env, we will use the global feature vector as a proxy for "Global Style"
            # OR better: use features_only=True if supported
            self.model = timm.create_model(model_name, pretrained=True, features_only=True).to(self.device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            self.model_loaded = True
        except Exception as e:
            print(f"[Glaze] Failed to load model {model_name} (features_only): {e}")
            try:
                # Fallback to standard
                self.model = timm.create_model(model_name, pretrained=True).to(self.device)
                self.model.eval()
                self.model_loaded = True
                self.features_only = False
                self.features_only = False
            except Exception:
                self.model_loaded = False
        else:
            self.features_only = True

        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.denormalize = T.Compose([
             T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
             T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
        ])

    def gram_matrix(self, input):
        a, b, c, d = input.size() 
        features = input.view(a * b, c * d) 
        G = torch.mm(features, features.t()) 
        return G.div(a * b * c * d)

    def generate_glaze(self, image: Image.Image, iterations=40, style_shift_target="van_gogh") -> Image.Image:
        """
        Shifts the style of the image to a target style (e.g., Van Gogh) 
        while preserving content (pixel distance).
        """
        if not self.model_loaded:
            return image

        original_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        delta = torch.zeros_like(original_tensor, requires_grad=True).to(self.device)
        
        optimizer = torch.optim.LBFGS([delta]) # LBFGS often better for style transfer, but SGD is safer for small step
        optimizer = torch.optim.SGD([delta], lr=0.05)

        # Generate "Alien Style" Target
        # In a real Glaze, we load a target style image.
        # Here, we simulate it by computing the style of a Random Noise Tensor 
        # or just randomizing the target Gram Matrix slightly.
        # Let's use a random tensor as "Alien Style"
        alien_style_input = torch.randn_like(original_tensor).to(self.device)
        
        alien_features = self.model(alien_style_input)
        if not isinstance(alien_features, list):
            alien_features = [alien_features]
        
        target_grams = [self.gram_matrix(f).detach() for f in alien_features]

        for i in range(iterations):
            def closure():
                optimizer.zero_grad()
                adv_input = original_tensor + delta
                
                features = self.model(adv_input)
                if not isinstance(features, list):
                    features = [features]
                
                style_loss = 0
                for ft, tg in zip(features, target_grams):
                    gm = self.gram_matrix(ft)
                    style_loss += F.mse_loss(gm, tg)
                
                # We want to MINIMIZE distance to Alien Style (Shift style)
                # But MAXIMIZE distance to Original Style? 
                # Glaze logic: Make it LOOK like Original to humans (Low Pixel MSE)
                # But LOOK like Alien to AI (Low Style Loss to Alien)
                
                # Constraint: Pixel L2/L-inf
                # Already implicit by delta initialization and clipping usually, but we add a penalty
                # content_loss = F.mse_loss(adv_input, original_tensor) # We want this small?
                # Actually we enforce this by clamping delta.
                
                total_loss = style_loss 
                total_loss.backward()
                return total_loss

            optimizer.step(closure)
            
            # Clip
            delta.data.clamp_(-0.05, 0.05) # Small perturbation for "invisible" cloak
            
        # Reconstruct
        adv_tensor = (original_tensor + delta).detach().cpu().squeeze(0)
        adv_tensor = self.denormalize(adv_tensor)
        adv_tensor = torch.clamp(adv_tensor, 0, 1)
        
        adv_image = T.ToPILImage()(adv_tensor)
        adv_image = adv_image.resize(image.size)
        
        return adv_image
