
import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from PIL import Image
import numpy as np
import cv2

class PrivacyCloak:
    """
    Protocol: "Visual Privacy" (Anti-ViT / Sparse Patch-Fool)
    Target: Vision Transformers (ViT) & Face Recognition
    Goal: Break Self-Attention on Facial Regions.
    """

    def __init__(self, model_name='deit_small_patch16_224', device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[PrivacyCloak] Initializing on {self.device}...")
        try:
            self.model = timm.create_model(model_name, pretrained=True).to(self.device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            self.model_loaded = True
        except Exception as e:
            print(f"[PrivacyCloak] Failed to load model {model_name}: {e}")
            self.model_loaded = False
            
        # Face Detector (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.denormalize = T.Compose([
             T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
             T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
        ])

    def detect_face_box(self, image: Image.Image):
        """Returns normalized (x, y, w, h) of the largest face or None."""
        # Convert PIL to CV2
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return None
        
        # Get largest face
        largest_face = max(faces, key=lambda r: r[2] * r[3])
        x, y, w, h = largest_face
        
        # Normalize to 0-1 range relative to image size
        H, W = img_cv.shape[:2]
        return (x/W, y/H, w/W, h/H)

    def generate_cloak(self, image: Image.Image, iterations=40, eps=4/255, alpha=1/255) -> Image.Image:
        """
        Optimizes a sparse noise patch on the face using Gradient-Guided Sensitivity.
        Algorithm: Sparse Attention Hijack.
        """
        if not self.model_loaded:
            return image

        original_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        original_tensor.requires_grad = True # We need grad w.r.t input for sensitivity map
        
        # 1. Compute Sensitivity Map (Gradient of Entropy w.r.t Input)
        self.model.zero_grad()
        outputs = self.model(original_tensor)
        probs = torch.softmax(outputs, dim=1)
        entropy = -1 * (probs * torch.log(probs + 1e-9)).sum() 
        entropy.backward()
        
        # Gradient Magnitude (Sum across RGB, Keep spatial)
        input_grad = original_tensor.grad.detach().abs().sum(dim=1).squeeze(0) # (224, 224)
        
        # 2. Create Sparse Mask (Top 2% sensitive pixels on FACE)
        mask = torch.zeros_like(original_tensor.detach()) # (1, 3, 224, 224)
        
        face_box = self.detect_face_box(image)
        H, W = 224, 224
        
        if face_box:
            fx, fy, fw, fh = face_box
            x1, y1 = int(fx*W), int(fy*H)
            x2, y2 = min(x1 + int(fw*W), W), min(y1 + int(fh*H), H)
            
            # Extract gradients only in face region
            face_grads = input_grad[y1:y2, x1:x2]
            
            # Find Top 2% Threshold
            k = int(0.02 * face_grads.numel())
            if k > 0:
                threshold = torch.topk(face_grads.reshape(-1), k).values[-1]
                # Create Local Mask
                local_mask = (face_grads >= threshold).float()
                # Expand to 3 channels and place in global mask
                mask[0, :, y1:y2, x1:x2] = local_mask.unsqueeze(0).repeat(3, 1, 1).to(self.device)
        else:
            # Fallback: Center Gradient Top 2%
            print("[PrivacyCloak] No face detected. Using center sensitivity.")
            center_grads = input_grad[H//4:3*H//4, W//4:3*W//4]
            k = int(0.02 * center_grads.numel())
            if k > 0:
                threshold = torch.topk(center_grads.reshape(-1), k).values[-1]
                local_mask = (center_grads >= threshold).float()
                mask[0, :, H//4:3*H//4, W//4:3*W//4] = local_mask.unsqueeze(0).repeat(3, 1, 1).to(self.device)

        # 3. PGD Optimization on Sparse Pixels
        # Clean up gradients from sensitivity step
        original_tensor.requires_grad = False
        original_tensor.grad = None
        
        delta = torch.zeros_like(original_tensor, requires_grad=True).to(self.device)
        optimizer = torch.optim.SGD([delta], lr=alpha)
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Perturb only masked pixels
            adv_img = original_tensor + (delta * mask)
            
            outputs = self.model(adv_img)
            
            # Objective: Maximize Entropy
            probs = torch.softmax(outputs, dim=1)
            loss = -1 * (probs * torch.log(probs + 1e-9)).sum() 
            
            loss.backward()
            optimizer.step()
            
            # Constraint: L_inf < 4/255 (Invisible)
            delta.data.clamp_(-eps, eps)
            
        # Reconstruct
        final_tensor = (original_tensor + (delta * mask)).detach().cpu().squeeze(0)
        final_tensor = self.denormalize(final_tensor)
        final_tensor = torch.clamp(final_tensor, 0, 1)
        
        adv_image = T.ToPILImage()(final_tensor)
        adv_image = adv_image.resize(image.size)
        return adv_image
