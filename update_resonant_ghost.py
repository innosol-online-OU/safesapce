
import os

file_path = "src/core/protocols/latent_cloak.py"
input_diversity_code = r'''
    def input_diversity(self, img_tensor, diversity_prob: float = 0.5):
        """
        Phase 16 [DIM]: Stochastic resizing and padding to build scale-robust noise.
        Matches 'Transform: Image is randomly resized (90%-110%) and padded.'
        """
        import torch
        if torch.rand(1).item() > diversity_prob:
            return img_tensor

        import torch.nn.functional as F
        B, C, H, W = img_tensor.shape
        
        # 1. Random resize factor (0.9 to 1.1)
        scale = 0.9 + (torch.rand(1).item() * 0.2)
        new_h, new_w = int(H * scale), int(W * scale)
        
        # 2. Resize
        resized = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # 3. Pad back to H, W (or crop if larger)
        if scale < 1.0:
            # Pad to fill
            pad_h = H - new_h
            pad_w = W - new_w
            # Random placement
            top = torch.randint(0, pad_h + 1, (1,)).item()
            left = torch.randint(0, pad_w + 1, (1,)).item()
            # F.pad format: (left, right, top, bottom)
            out = F.pad(resized, (left, pad_w - left, top, pad_h - top), value=0.0)
        else:
            # Crop to fit
            diff_h = new_h - H
            diff_w = new_w - W
            # Safety check for exact match
            if diff_h == 0 and diff_w == 0:
                 return resized
            
            top = torch.randint(0, diff_h + 1 if diff_h > 0 else 1, (1,)).item() if diff_h > 0 else 0
            left = torch.randint(0, diff_w + 1 if diff_w > 0 else 1, (1,)).item() if diff_w > 0 else 0
            out = resized[:, :, top:top+H, left:left+W]
                 
        return out
'''

protect_phantom_code = r'''
    def protect_phantom(self, image_path: str, strength=50, retries=1, user_mask=None, 
                       targeting_intensity: float = 3.0, resolution: str = "384", 
                       background_intensity: float = 0.2, decay_factor: float = 1.0, 
                       diversity_prob: float = 0.5):
        """
        Phase 16: Resonant Ghost (Pure Phantom).
        Uses MI-FGSM (Momentum) + DIM (Input Diversity) + SigLIP (Critic)
        to generate robust, structural adversarial noise.
        """
        self._load_optimizer()
        import torch.nn.functional as F
        import torch.optim as optim
        import numpy as np
        
        # 1. Config & Preprocessing
        # -------------------------
        # Steps typically 40-100 for proper convergence
        num_steps = int(40 + (strength/100)*60) 
        
        # Alpha (Step Size)
        alpha = (1.5 + (strength / 50.0)) / 255.0
        
        res_val = 512 # Default working resolution
        if resolution != "Original":
            try:
                res_val = int(resolution)
            except: 
                pass
        
        # Load Image
        orig_pil = Image.open(image_path).convert("RGB")
        w_orig, h_orig = orig_pil.size
        
        # Resize for optimization
        if resolution != "Original":
            scale = res_val / max(w_orig, h_orig)
            w_work, h_work = int(w_orig * scale), int(h_orig * scale)
            work_pil = orig_pil.resize((w_work, h_work), Image.LANCZOS)
        else:
            work_pil = orig_pil
            w_work, h_work = w_orig, h_orig
            
        # 2. JND & Targeting Mask
        # ---------------------
        img_tensor = torch.from_numpy(np.array(work_pil)).to(self.device).float().permute(2,0,1).unsqueeze(0) / 255.0
        
        # Base JND
        jnd_tensor_base = self.compute_jnd_mask(img_tensor, strength/100.0)
        
        # Convert to Numpy
        jnd_numpy = jnd_tensor_base.detach().cpu().numpy()[0].transpose(1, 2, 0)
        jnd_numpy = jnd_numpy * background_intensity 
        
        # Targeting Boost
        target_mask = np.zeros((h_work, w_work), dtype=np.float32)
        
        # A. Neural Targeting
        img_cv = cv2.cvtColor(np.array(work_pil), cv2.COLOR_RGB2BGR)
        faces = self.face_analysis.get(img_cv)
        if faces:
            for face in faces:
                box = face.bbox.astype(int)
                x1, y1, x2, y2 = max(0, box[0]-10), max(0, box[1]-10), min(w_work, box[2]+10), min(h_work, box[3]+10)
                target_mask[y1:y2, x1:x2] = 1.0
                
        # B. User Mask
        if user_mask is not None:
             um_pil = Image.fromarray((user_mask * 255).astype(np.uint8))
             um_resized = um_pil.resize((w_work, h_work), Image.NEAREST)
             um_arr = np.array(um_resized).astype(np.float32) / 255.0
             target_mask = np.maximum(target_mask, um_arr)
             
        # Apply boost
        target_mask_3d = np.expand_dims(target_mask, axis=2)
        jnd_limit = jnd_numpy + (jnd_numpy * target_mask_3d * targeting_intensity)
        jnd_tensor = torch.from_numpy(jnd_limit).to(self.device).float().permute(2,0,1).unsqueeze(0)
        
        # 3. Optimization Setup
        # ---------------------
        delta = torch.zeros_like(img_tensor).to(self.device).requires_grad_(True)
        momentum = torch.zeros_like(img_tensor).to(self.device)
        
        critic_model, mean, std = self.siglip
        
        # Anchor Embeddings
        with torch.no_grad():
             img_384 = F.interpolate(img_tensor, size=(384, 384), mode='bilinear')
             norm_img = (img_384 - mean) / std
             orig_features = critic_model(norm_img)
             orig_features = orig_features / orig_features.norm(dim=-1, keepdim=True)
            
        logger.info(f"[ResonantGhost] Steps: {num_steps} | Alpha: {alpha*255:.2f}/255 | Decay: {decay_factor} | DIM: {diversity_prob}")
        
        # Heatmap Save
        try:
            mask_max = jnd_limit.max()
            debug_mask = jnd_limit / (mask_max + 1e-8)
            debug_mask_np = (debug_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(debug_mask_np, cv2.COLORMAP_JET)
            os.makedirs("/app/uploads", exist_ok=True)
            cv2.imwrite("/app/uploads/debug_jnd_heatmap.png", heatmap)
        except Exception as e:
            logger.warning(f"[LatentCloak] Heatmap failed: {e}")

        # 4. Optimization Loop (MI-FGSM)
        # ------------------------------
        for i in range(num_steps):
            if delta.grad is not None:
                delta.grad.zero_()
            
            # DIM
            adv_base = img_tensor + delta
            adv_div = self.input_diversity(adv_base, diversity_prob)
            
            # Forward
            div_384 = F.interpolate(adv_div, size=(384, 384), mode='bilinear')
            norm_div = (div_384 - mean) / std
            adv_features = critic_model(norm_div)
            adv_features = adv_features / adv_features.norm(dim=-1, keepdim=True)
            
            # Loss: Minimize Cos Similarity (Push Away)
            loss = F.cosine_similarity(adv_features, orig_features).mean()
            loss.backward()
            
            # Momentum
            if delta.grad is None: continue
            grad = delta.grad.data
            grad_norm = torch.norm(grad, p=1)
            grad = grad / (grad_norm + 1e-10)
            momentum = decay_factor * momentum + grad
            
            # Update (Gradient Descent on Sim = Gradient Ascent on Diff)
            # Minimize Sim -> Go opposite to gradient
            delta.data = delta.data - alpha * momentum.sign()
            
            # Constraints
            delta.data = torch.max(torch.min(delta.data, jnd_tensor), -jnd_tensor)
            delta.data = torch.clamp(img_tensor + delta.data, 0, 1) - img_tensor
            
            if i % 10 == 0:
                print(f"Ghost Step {i}/{num_steps} | Sim: {loss.item():.4f}", flush=True)

        # 5. Finalize
        with torch.no_grad():
            final_delta = delta.data
            if resolution != "Original":
                final_delta = F.interpolate(final_delta, size=(h_orig, w_orig), mode='bicubic', align_corners=False)
                final_image_tensor = (original_tensor + final_delta).clamp(0, 1)
            else:
                final_image_tensor = (original_tensor + final_delta).clamp(0, 1)

        final_np = final_image_tensor.cpu().permute(0, 2, 3, 1).numpy()[0]
        final_pil = Image.fromarray((final_np * 255).astype(np.uint8))
        return final_pil
'''

# Read file
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find start of protect_phantom
start_idx = -1
for i, line in enumerate(lines):
    if "def protect_phantom(self," in line:
        start_idx = i
        break

if start_idx == -1:
    print("Could not find protect_phantom definition to replace.")
    exit(1)

print(f"Truncating at line {start_idx} and appending new implementation...")

# Keep comments/decorators above? No, previous implementation didn't have decorators other than docstring inside.
# Truncate lines
lines = lines[:start_idx]

# Write back
with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(lines)
    f.write("\n")
    f.write(input_diversity_code)
    f.write("\n")
    f.write(protect_phantom_code)

print("LatentCloak updated successfully.")
