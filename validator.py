
import os
import torch
import numpy as np
from PIL import Image
import cv2
import warnings

# Suppress heavy warnings
warnings.filterwarnings("ignore")

try:
    import lpips
    from deepface import DeepFace
    from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
except ImportError:
    print("Warning: Validator dependencies missing (lpips, deepface, transformers).")

class Validator:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"[Validator] Initializing on {self.device}...")
        
        self.lpips_loss = None
        self.clip_model = None
        self.clip_processor = None
        
        try:
            # 1. LPIPS (Visual Perception)
            # AlexNet variant is standard for perceptual similarity
            self.lpips_loss = lpips.LPIPS(net='alex').to(self.device)
            
            # 2. CLIP (Grok Simulation)
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
            
            print("[Validator] Models loaded.")
        except Exception as e:
            print(f"[Validator] Setup failed: {e}")

    def validate(self, original_path, protected_path, identity_text="A photo of a face"):
        print(f"\n--- Red Team Verification: {os.path.basename(protected_path)} ---")
        
        results = {}
        
        try:
            # Load Images
            img_orig = Image.open(original_path).convert("RGB")
            img_prot = Image.open(protected_path).convert("RGB")
            
            # 1. Visual Quality (LPIPS)
            # Expect < 0.1 for "Invisible"
            val_lpips = self.check_lpips(img_orig, img_prot)
            print(f"1. LPIPS (Visual Diff): {val_lpips:.4f} (Target < 0.1)")
            results['lpips'] = val_lpips
            results['invisible'] = val_lpips < 0.1
            
            # 2. Identity Check (DeepFace)
            # Expect Verified == False
            verified = self.check_deepface(original_path, protected_path)
            print(f"2. DeepFace Verification: {verified} (Target: False)")
            results['identity_broken'] = not verified
            
            # 3. Semantic Check (CLIP)
            # Check drop in similarity to identity text
            # e.g. "Elon Musk"
            drop = self.check_clip_drop(img_orig, img_prot, identity_text)
            print(f"3. CLIP Score Drop: {drop:.2f}% (Target > 30%)")
            results['clip_drop'] = drop
            results['semantically_broken'] = drop > 30.0
            
            # Summary
            if results['invisible'] and results['identity_broken']:
                print("✅ PROJECT INVISIBLE: SUCCESS")
                return True
            else:
                print("❌ PROJECT INVISIBLE: FAILED")
                return False
                
        except Exception as e:
            print(f"[Validator] Runtime Error: {e}")
            return False

    def check_lpips(self, img1_pil, img2_pil):
        # Convert to tensor [-1, 1]
        t1 = self._to_tensor(img1_pil)
        t2 = self._to_tensor(img2_pil)
        with torch.no_grad():
            dist = self.lpips_loss(t1, t2)
        return dist.item()

    def check_deepface(self, path1, path2):
        try:
            # DeepFace.verify returns dict
            # using ArcFace usually
            result = DeepFace.verify(img1_path=path1, img2_path=path2, model_name="ArcFace", enforce_detection=False)
            return result['verified']
        except Exception as e:
            print(f"DeepFace Error: {e}")
            return True # Fail open (assume verified if error)

    def check_clip_drop(self, img1, img2, text):
        inputs1 = self.clip_processor(text=[text], images=img1, return_tensors="pt", padding=True).to(self.device)
        inputs2 = self.clip_processor(text=[text], images=img2, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs1 = self.clip_model(**inputs1)
            outputs2 = self.clip_model(**inputs2)
            
            # Image-Text similarity
            sim1 = outputs1.logits_per_image.item()
            sim2 = outputs2.logits_per_image.item()
            
        # Calc percent drop
        # Logits can be negative/positive, but usually large positive. CLIP uses cosine.
        # Let's use cosine similarity directly if logits are messy? 
        # OpenAI CLIP logits are scaled cosine.
        # Let's treat raw score drop.
        
        if sim1 == 0: return 0.0
        pct_drop = ((sim1 - sim2) / sim1) * 100
        return pct_drop

    def _to_tensor(self, img):
        # Resize to 256 for consistent LPIPS? Or native.
        # LPIPS expects (N,3,H,W) and [-1, 1]
        t = torch.tensor(np.array(img)).permute(2,0,1).float() / 255.0
        t = (t * 2.0) - 1.0
        return t.unsqueeze(0).to(self.device)
