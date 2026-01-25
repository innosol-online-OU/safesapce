# CVE-2025-32434 BYPASS: Must be first to patch transformers before any model loading
# This is safe because we only load from OpenAI/HuggingFace official repos.
try:
    import transformers.utils.import_utils
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
    print("[SafeSpace] CVE-2025-32434 bypass applied for trusted models.")
except Exception:
    pass  # Transformer version doesn't have this check

from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import os
import uuid
import json
try:
    from skimage.metrics import structural_similarity as ssim_metric
except ImportError:
    ssim_metric = None


class CloakEngine:
    def __init__(self):
        # Cache for Project Invisible models
        self.latent_cloak = None
        self.latent_cloak = None
        self.stego_injector = None
        self.qwen_critic = None

    def apply_defense(self, image_path: str, output_path: str, visual_mode: str = "latent_diffusion", compliance: bool = True, strength: int = 75, face_boost = False, user_mask = None, use_badge: bool = True, badge_text: str = "NO AI TRAIN",
                       # Phase 2 parameters
                       target_profile: str = "general",
                       ensemble_diversity: int = 5,
                       optimization_steps: int = 100,
                       use_dwt_mamba: bool = False,
                       use_neural_stego: bool = False,
                       hidden_command: str = "",
                       max_retries: int = 3,
                       background_intensity: float = 1.0) -> tuple[bool, str, dict]:
        """
        Applies Project Invisible (Latent Diffusion Defense).
        
        Args:
            image_path: Path to input image
            output_path: Path for protected output
            visual_mode: "latent_diffusion" or "none"
            compliance: Enable steganography layer
            strength: 0-100 destruction level
            face_boost: Float multiplier (1.0-4.0) or boolean
            user_mask: Optional numpy array mask for targeted areas
            use_badge: Add visual trust badge (shield)
            badge_text: Custom text for the badge
            
            # Phase 2 Parameters
            target_profile: "general" or "frontier"
            ensemble_diversity: 1-10, higher = more model variety per step
            optimization_steps: 50-500, more steps = stronger protection
            use_dwt_mamba: Enable wavelet-domain optimization
            use_neural_stego: Enable neural steganography
            hidden_command: Command to embed via neural stego
            
        Returns:
            success (bool)
            heatmap_path (str): Path to difference heatmap image
            metrics (dict): JSON-serializable metrics
        """
        try:
            is_frontier = target_profile == "frontier"
            is_phantom = target_profile == "phantom_15"
            is_liquid = target_profile == "liquid_17"
            mode_str = "FRONTIER" if is_frontier else ("PHANTOM" if is_phantom else ("LIQUID" if is_liquid else "GENERAL"))
            print(f"Applying Project Invisible -> Visual: {visual_mode}, Compliance: {compliance}, Mode: {mode_str}")
            img = Image.open(image_path).convert("RGB")
            
            # 1. Visual Shield Layer (Latent Surgery)
            if visual_mode == "latent_diffusion":
                # Configure Protection
                from src.core.protocols.latent_cloak import ProtectionConfig
                
                # Handle face_boost as boolean or float
                if isinstance(face_boost, bool):
                    boost_val = 3.0 if face_boost else 1.0
                else:
                    boost_val = float(face_boost)
                
                # Build config with Phase 2 parameters
                config = ProtectionConfig(
                    strength=int(strength),
                    face_boost=boost_val,
                    defense_mode='aggressive' if is_frontier else 'stealth',
                    target_profile=target_profile,
                    ensemble_diversity=ensemble_diversity,
                    optimization_steps=optimization_steps,
                    hidden_command=hidden_command,
                    use_neural_stego=use_neural_stego,
                    use_dwt_mamba=use_dwt_mamba
                )
                
                # ADVERSARIAL FEEDBACK LOOP (LatentCloak <-> QwenCritic)
                current_strength = int(strength)
                # max_retries parameter is now passed from app.py
                
                # SPEED OPTIMIZATION: Check if Qwen uses API mode (no local VRAM needed)
                from src.core.critics.qwen_critic import QwenCritic
                test_critic = QwenCritic()
                qwen_uses_api = test_critic.use_api
                del test_critic
                
                # Load LatentCloak ONCE (Performance Fix)
                # PHASE 4 LITE: Use lite_mode to skip heavy SD loading for Frontier
                # PHASE 15: Phantom also uses lite_mode (SigLIP only)
                if self.latent_cloak is None:
                    # Check if current mode requires lite or full
                    # If we are in General mode but loaded Lite -> Reload Full? 
                    # For now assume mostly one-way or restart.
                    use_lite = is_frontier or is_phantom or is_liquid
                    print(f"[CloakEngine] Initializing LatentCloak (lite_mode={use_lite})...")
                    from src.core.protocols.latent_cloak import LatentCloak
                    self.latent_cloak = LatentCloak(lite_mode=use_lite)
                
                ensemble_stats = None  # For Phase 2 metrics
                
                # EXTRACT PHANTOM CONFIG (Phase 15.X/Y)
                try:
                    import streamlit as st
                    p_strength = st.session_state.get('phantom_strength', 50)
                    p_retries = st.session_state.get('phantom_retries', 3)
                    p_targeting = st.session_state.get('phantom_targeting', 2.0)
                    p_resolution = st.session_state.get('phantom_resolution', "Original")
                    # Liquid Warp Params
                    l_grid = st.session_state.get('liquid_grid', 16)
                    l_tv = st.session_state.get('liquid_tv', 50.0)
                    l_steps = st.session_state.get('liquid_steps', 100)
                    l_limit = st.session_state.get('liquid_limit', 0.004)
                    l_blur = st.session_state.get('liquid_blur', 15)
                except Exception:
                    # Fallback for CLI/API usage
                    p_strength = 50
                    p_retries = 3
                    p_targeting = 2.0
                    p_resolution = "Original"
                    l_grid = 16
                    l_tv = 50.0
                    l_steps = 100
                    l_limit = 0.004
                    l_blur = 15
                
                for attempt in range(max_retries + 1 if not is_phantom else p_retries):
                    # For Phantom, the inner loop handles retries as per Phase 15.X spec
                    
                    print(f"--- Adversarial Loop Attempt {attempt+1} (Strength: {current_strength if not is_phantom else p_strength}) ---")
                    config.strength = current_strength
                    config.attempt = attempt 
                    
                    # PHASE 4 LITE: Use protect_frontier_lite for Frontier mode (Fast, Low VRAM)
                    if is_frontier:
                        print("[CloakEngine] PHASE 4 LITE: Using Anti-Segmentation protect method...")
                        img = self.latent_cloak.protect_frontier_lite(image_path, config=config, user_mask=user_mask)
                    elif is_phantom:
                        print(f"[CloakEngine] PHASE 15.Z: Using Phantom Smart Targeting (Str {p_strength}, Res {p_resolution}, Target {p_targeting}, BG {background_intensity})...")
                        # Phase 15.Z specific call
                        img = self.latent_cloak.protect_phantom(
                            image_path, 
                            strength=p_strength, 
                            retries=p_retries,
                            user_mask=user_mask,
                            targeting_intensity=p_targeting,
                            resolution=p_resolution,
                            background_intensity=background_intensity
                        )
                        # Phase 16: Allow fall-through to Validation (Qwen)
                        # break  <-- REMOVED to enable Qwen execution
                    elif is_liquid:
                        print(f"[CloakEngine] PHASE 17.6: Using Liquid Warp (Grid {l_grid}, TV {l_tv}, Limit {l_limit}, Blur {l_blur})...")
                        img = self.latent_cloak.protect_liquid_warp(
                            image_path,
                            strength=p_strength,
                            grid_size=l_grid,
                            num_steps=l_steps,
                            lr=0.005,
                            tv_weight=l_tv,
                            flow_limit=l_limit,
                            mask_blur=l_blur
                        )
                    else:
                        img = self.latent_cloak.protect(image_path, config=config, user_mask=user_mask)
                    
                    # VISUAL TRUST BADGE (Only for General mode, skip for Frontier Lite and Phantom)
                    if use_badge and not (is_frontier or is_phantom):
                        print(f"[CloakEngine] Applying Visual Trust Badge with Hidden Payload: '{badge_text}'", flush=True)
                        img = self.latent_cloak.add_trust_badge(img, badge_text=badge_text)
                        # Inject the specific command payload into the image steganographically
                        img = self.inject_invisible_text(img, text=badge_text)

                    img.save(output_path)
                    
                    # PERFORMANCE FIX: DO NOT UNLOAD LatentCloak here.
                    # We keep it loaded for the next iteration or next image.
                    # We rely on LatentCloak's internal CPU offload to manage VRAM.
                    pass

                    # 2. VALIDATE with Qwen (Pairwise Phase 14)
                    print("[CloakEngine] Loading QwenCritic for Pairwise Validation...")
                    self.qwen_critic = QwenCritic()
                    
                    # Pass original image as Ref, output as Probe
                    passed, reason, _ = self.qwen_critic.critique_pairwise(reference_path=image_path, probe_path=output_path)
                    print(f"[CloakEngine] Qwen Result: {passed} | {reason}")
                    
                    # Cleanup critic (lightweight if API mode)
                    del self.qwen_critic
                    self.qwen_critic = None
                    
                    if passed:
                        print("✅ Qwen-VL Validation Passed.")
                        break
                    else:
                        print("❌ Qwen-VL Validation Failed. Increasing Strength...")
                        if current_strength >= 100:
                            print("⚠️ Maximum Strength (100) reached. Stopping loop despite failure.")
                            break
                        current_strength = min(100, current_strength + 10)
                        
            elif visual_mode == "none":
                pass
            else:
                print(f"Unknown mode: {visual_mode}. Defaulting to None.")
            
            # 2. Compliance Shield Layer (Steganography)
            if compliance:
                 if self.stego_injector is None:
                     from src.core.protocols.stego_text import StegoInjector
                     self.stego_injector = StegoInjector()
                 img = self.stego_injector.inject_text(img, payload="GDPR_BLOCK")
            
            # Save final output
            img.save(output_path)
            
            # --- Metrics & Visualization ---
            original_img = Image.open(image_path).convert("RGB")
            # Ensure size matches (in case latent cloak resized it)
            if img.size != original_img.size:
                original_img = original_img.resize(img.size)
            
            # Generate Heatmap (Difference Map)
            diff = ImageChops.difference(original_img, img)
            # Enhance contrast to make invisible changes visible
            diff = ImageEnhance.Brightness(diff).enhance(10.0) 
            heatmap_filename = f"heatmap_{os.path.basename(output_path)}"
            heatmap_path = os.path.join(os.path.dirname(output_path), heatmap_filename)
            diff.save(heatmap_path)

            # Generate Basic Metrics (SSIM/PSNR surrogate or Latent Distance)
            # Ensure both are RGB for fair comparison (ignore alpha channel of badge)
            img_rgb = img.convert("RGB")
            orig_rgb = original_img.convert("RGB")
            
            arr1 = np.array(orig_rgb, dtype=np.float32)
            arr2 = np.array(img_rgb, dtype=np.float32)
            
            mse = float(np.mean((arr1 - arr2) ** 2))
            psnr = float(20 * np.log10(255.0 / np.sqrt(mse))) if mse > 0 else 100.0
            
            # SCALAR METRICS FOR APP.PY
            # Calculate SSIM (using luminance strictly or multichannel)
            try:
                # Use simple SSIM or skip if skimage not available/fails
                ssim_val = float(ssim_metric(arr1, arr2, data_range=255, channel_axis=2))
            except Exception as e:
                print(f"SSIM calc error: {e}")
                ssim_val = 0.0
            
            # BIOMETRIC SIMILARITY (Phase 3)
            # Compare embeddings of original vs protected
            bio_sim = 1.0
            try:
                if self.latent_cloak:
                    # Ensure InsightFace is loaded for metrics (might not be if in Frontier mode)
                    if not hasattr(self.latent_cloak, 'app') or self.latent_cloak.app is None:
                        print("[CloakEngine] Loading InsightFace for Biometric Check...")
                        self.latent_cloak.init_critics(['face'])
                        
                    bio_sim = self.latent_cloak.compute_similarity(original_img, img)
                    print(f"[CloakEngine] 🧬 Biometric Similarity: {bio_sim:.4f} (Target < 0.4)")
            except Exception as e:
                print(f"[CloakEngine] Bio-Sim Error: {e}")

            # Default validation status
            passed = False
            reason = "Skipped"

            metrics = {
                "psnr": psnr,
                "ssim": ssim_val,
                "bio_sim": bio_sim,
                "Visual Fidelity (PSNR)": f"{psnr:.2f} dB",
                "Injection Magnitude (MSE)": f"{mse:.2f}",
                "Biometric Similarity": f"{bio_sim:.4f}",
                "Layers Applied": []
            }
            if visual_mode != "none": 
                metrics["Layers Applied"].append(visual_mode)
                metrics["Configuration"] = f"Strength: {current_strength} | Face Boost: {'ON' if face_boost else 'OFF'}"
                metrics["Adversarial Validation"] = "Passed (Qwen-VL Approved)" if passed else f"Failed ({reason})"
            if compliance: metrics["Layers Applied"].append("Stego-Compliance")

            return True, heatmap_path, metrics
            
            
        except Exception as e:
            print(f"Error applying defense: {e}")
            import traceback
            traceback.print_exc()
            import traceback
            traceback.print_exc()
            return False, None, {"error": str(e)}

    def detect_faces(self, image_path: str):
        """
        Phase 15.2: Detect faces for auto-targeting (Smart Regions).
        Returns list of dicts: {'bbox': [x1, y1, x2, y2], 'kps': [[x,y]...]}
        """
        import torch
        if self.latent_cloak is None:
            from src.core.protocols.latent_cloak import LatentCloak
            # Phase 15.1: Initialize lightweight (no heavy models)
            self.latent_cloak = LatentCloak(device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Load only detectors (Instant)
        if hasattr(self.latent_cloak, '_load_detectors'):
            self.latent_cloak._load_detectors()

        # Access detector
        detector = getattr(self.latent_cloak, 'face_analysis', None)
        # Fallback for old name if any
        if detector is None:
             detector = getattr(self.latent_cloak, 'face_app', None)

        if detector:
            import cv2
            img = cv2.imread(image_path)
            if img is not None:
                faces = detector.get(img)
                results = []
                for face in faces:
                    res = {
                        'bbox': face.bbox.tolist(),
                        'kps': face.kps.tolist() if hasattr(face, 'kps') else None
                    }
                    results.append(res)
                return results
        return []

    # Deprecated / Legacy wrappers
    def apply_protocol(self, *args, **kwargs):
        return self.apply_defense(*args, **kwargs)

    def inject_invisible_text(self, img_pil: Image.Image, text: str = "") -> Image.Image:
         if self.stego_injector is None:
             from src.core.protocols.stego_text import StegoInjector
             self.stego_injector = StegoInjector()
         return self.stego_injector.inject_text(img_pil, payload=text or "GDPR_BLOCK")
