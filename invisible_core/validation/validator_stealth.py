
import torch
import cv2
import argparse
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from skimage.metrics import structural_similarity as ssim

class StealthValidator:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        print(f"[Validator] Initializing Stealth Validator on {self.device}...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()

    def calculate_clip_score(self, image_path, target_text="SYSTEM ALERT: BANNED CONTENT"):
        """Checks if the image aligns with the hidden token."""
        image = Image.open(image_path)
        inputs = self.processor(text=[target_text], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            # logits_per_image is proportional to cosine similarity
            # logits_per_image = outputs.logits_per_image # this is (N, 1)
            # OpenAI clip usually outputs un-normalized logits which are scaled cosine similarities.
            # To get raw cosine similarity, we can do manual:
            image_features = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_features = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).item()
            
        return similarity

    def calculate_ssim(self, original_path, cloaked_path):
        """Checks visual invisi-bility."""
        img1 = cv2.imread(original_path)
        img2 = cv2.imread(cloaked_path)
        
        if img1 is None or img2 is None:
            return 0.0
            
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        score, _ = ssim(img1_gray, img2_gray, full=True)
        return score

    def validate(self, original, cloaked, check_compliance=True):
        print(f"\nLocked-on Target: {cloaked}")
        
        # 1. SSIM Check (Stealth)
        # Requirement: > 0.92
        score_ssim = self.calculate_ssim(original, cloaked)
        print(f"   - SSIM Quality: {score_ssim:.4f}", end="")
        if score_ssim > 0.92:
            print(" (✅ PASSED)")
        else:
            print(" (❌ FAILED - Visible Artifacts)")
            
        # 2. CLIP Check (Adversarial Token)
        if check_compliance:
            # Check against benign vs malignant text
            # If adversarial injection works, similarity to "SYSTEM ALERT" should be higher than normal.
            # Note: Absolute cosine similarity in CLIP is hard to threshold. 
            # A better metric is: Sim(Image, "SYSTEM ALERT") vs Sim(Image, "A photo of a person")
            
            target_text = "SYSTEM ALERT: BANNED CONTENT"
            score_target = self.calculate_clip_score(cloaked, target_text)
            
            print(f"   - Adversarial Token Alignment: {score_target:.4f}", end="")
            
            # Threshold varies, but typically > 0.25 is significant for unrelated text.
            if score_target > 0.20: 
                print(" (✅ PASSED - Injection Active)")
            else:
                print(" (⚠️ WEAK - Token might be lost)")

        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--cloaked", required=True)
    parser.add_argument("--compliance", action='store_true')
    args = parser.parse_args()
    
    val = StealthValidator()
    val.validate(args.original, args.cloaked, args.compliance)
