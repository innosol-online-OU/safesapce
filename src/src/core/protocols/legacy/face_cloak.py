
import torch
import sys
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Add external repo to path to import their modules
# Assuming src/external/adv_makeup is the clone dir
EXTERNAL_PATH = os.path.join(os.getcwd(), 'src', 'external', 'adv_makeup')
sys.path.append(EXTERNAL_PATH)

try:
    from networks import Encoder, Decoder
except ImportError:
    print(f"[FaceCloak] Warning: Could not import Adv-Makeup networks from {EXTERNAL_PATH}")

from src.core.util.face_masker import FaceMasker

class FaceCloak:
    def __init__(self, weights_path="models/adv_makeup/AdvMakeup_Gen.pth", device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[FaceCloak] Initializing on {self.device}")
        
        self.masker = FaceMasker()
        
        # Load Model
        # Using dimensions from config.py default (input_dim=3)
        self.enc = Encoder(input_dim=3).to(self.device)
        self.dec = Decoder(output_dim=3).to(self.device)
        
        self.weights_path = weights_path
        self.model_loaded = False
        self.load_weights()
        
        self.preprocess = T.Compose([
            T.Resize((256, 256)), # Standardize for inference
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_weights(self):
        if os.path.exists(self.weights_path):
            try:
                state_dict = torch.load(self.weights_path, map_location=self.device)
                # Check structure
                if 'enc' in state_dict and 'dec' in state_dict:
                     self.enc.load_state_dict(state_dict['enc'])
                     self.dec.load_state_dict(state_dict['dec'])
                else:
                     # Fallback if user saved state_dicts directly into one file in a different way
                     # or if it's just the generator state dict (less likely given usage)
                     print("[FaceCloak] Warning: Unexpected weight format. Expecting {'enc': ..., 'dec': ...}")
                
                self.enc.eval()
                self.dec.eval()
                self.model_loaded = True
                print("[FaceCloak] Weights loaded successfully.")
            except Exception as e:
                print(f"[FaceCloak] Failed to load weights: {e}")
        else:
             print(f"[FaceCloak] Weights not found at {self.weights_path}. Running in dummy mode (Initialized weights).")
             # We allow running without weights for testing pipeline flow, 
             # though output will be garbage (untrained).

    def protect(self, image: Image.Image) -> Image.Image:
        """
        Applies Adv-Makeup generator output ONLY to the face region.
        """
        # A. Mask
        # FaceMasker expects PIL, returns (H, W) 0-1 float
        mask = self.masker.get_mask(image) 
        
        # B. Inference
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
             content = self.enc(img_tensor)
             noise_tensor = self.dec(*content)
             
        # Post-process noise matches image size
        noise_img = noise_tensor.squeeze(0).cpu()
        # Denormalize [-1, 1] -> [0, 1]
        noise_img = noise_img * 0.5 + 0.5 
        noise_img = torch.clamp(noise_img, 0, 1)
        
        noise_pil = T.ToPILImage()(noise_img).resize(image.size)
        noise_np = np.array(noise_pil).astype(np.float32) / 255.0
        
        img_np = np.array(image).astype(np.float32) / 255.0
        
        # Ensure mask shape matches image (H, W, 1) or (H, W) -> broadcast
        if len(mask.shape) == 2:
            mask_exp = np.expand_dims(mask, axis=-1)
        else:
            mask_exp = mask

        # C. Blend: Original * (1-Mask) + Noise * Mask
        # If image is RGBA, handle alpha? Assuming RGB for now.
        if img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]
        
        final = (img_np * (1 - mask_exp)) + (noise_np * mask_exp)
        
        # Clip just in case
        final = np.clip(final, 0, 1)
        return Image.fromarray((final * 255).astype(np.uint8))
