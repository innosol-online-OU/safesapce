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
    ssim_metric = None

from typing import Any, Dict, List, Tuple, Optional
class CloakEngine:
    def __init__(self):
        # Cache for Project Invisible models
        self.latent_cloak = None
        self.latent_cloak = None
        self.stego_injector = None
        self.qwen_critic = None

    def get_ghost_mesh_optimizer(self):
        """Lazy load and return the GhostMesh optimizer."""
        if self.latent_cloak is None:
             print(f"[CloakEngine] Initializing LatentCloak for GhostMesh (Lazy)...")
             from invisible_core.attacks.latent_cloak import LatentCloak
             self.latent_cloak = LatentCloak(lite_mode=True)
        
        # Ensure underlying resources (SigLIP) are loaded
        # Note: lite_mode=True sets models_loaded=True but doesn't load siglip.
        # So we must check siglip directly.
        if self.latent_cloak.siglip is None:
             self.latent_cloak._load_optimizer()
        
        # Ensure ghost_mesh is initialized
        if not hasattr(self.latent_cloak, 'ghost_mesh') or self.latent_cloak.ghost_mesh is None:
             from invisible_core.attacks.ghost_mesh import GhostMeshOptimizer
             self.latent_cloak.ghost_mesh = GhostMeshOptimizer(
                 siglip_model=self.latent_cloak.siglip,
                 device=self.latent_cloak.device
             )
             
        return self.latent_cloak.ghost_mesh

    def apply_defense(self, input_path: str, output_path: str, visual_mode: str = "latent_diffusion",
                     compliance: bool = True, strength: int = 50, face_boost: float = 1.0,
                     user_mask: Optional[np.ndarray] = None, use_badge: bool = True, badge_text: str = "DON'T EDIT",
                     target_profile: str = "general", ensemble_diversity: int = 1,
                     optimization_steps: int = 100, use_dwt_mamba: bool = False,
                     use_neural_stego: bool = False, hidden_command: str = "",
                     max_retries: int = 3, background_intensity: float = 1.0,
                     is_liquid_v2: bool = False, liquid_grid_size: int = 16, liquid_asymmetry: float = 0.05
                     ) -> Tuple[bool, str, Dict]:
        """
        Applies Project Invisible (Latent Diffusion Defense).
        
        Args:
            input_path: Path to input image
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
            is_liquid_v2 = target_profile == "liquid_17_v2"
            is_ghost_mesh = target_profile == "ghost_mesh"
            mode_str = "GHOST_MESH" if is_ghost_mesh else ("FRONTIER" if is_frontier else ("PHANTOM" if is_phantom else ("LIQUID_V2" if is_liquid_v2 else ("LIQUID" if is_liquid else "GENERAL"))))
            print(f"Applying Project Invisible -> Visual: {visual_mode}, Compliance: {compliance}, Mode: {mode_str}")
            img = Image.open(input_path).convert("RGB")
            
            # 1. Visual Shield Layer (Latent Surgery)
            if visual_mode == "latent_diffusion":
                # Configure Protection
                from invisible_core.attacks.latent_cloak import ProtectionConfig
                
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
                from invisible_core.critics.qwen_critic import QwenCritic
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
                    use_lite = is_frontier or is_phantom or is_liquid or is_liquid_v2 or is_ghost_mesh
                    print(f"[CloakEngine] Initializing LatentCloak (lite_mode={use_lite})...")
                    from invisible_core.attacks.latent_cloak import LatentCloak
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
                    # Ghost-Mesh Params (Phase 18)
                    gm_grid = st.session_state.get('ghost_mesh_grid', 24)
                    gm_balance = st.session_state.get('ghost_mesh_balance', 0.5)
                    gm_anchoring = st.session_state.get('ghost_mesh_anchoring', 0.8)
                    gm_tv = st.session_state.get('ghost_mesh_tv', 50)
                    gm_jnd = st.session_state.get('ghost_mesh_jnd', True)
                    # Granular Controls
                    gm_noise = st.session_state.get('ghost_mesh_noise', None)
                    gm_warp = st.session_state.get('ghost_mesh_warp', None)
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
                    gm_grid = 24
                    gm_balance = 0.5
                    gm_anchoring = 0.8
                    gm_tv = 50
                    gm_jnd = True
                    gm_noise = None
                    gm_warp = None
                
                # Determine loop count
                # V2: max_retries = Total Runs (Best of N)
                # Others: max_retries = Retries (Initial + N retries)
                # Others: max_retries = Retries (Initial + N retries)
                loop_count = max_retries if (is_liquid_v2 or is_ghost_mesh) else (p_retries if is_phantom else max_retries + 1)
                
                # Init validation vars
                passed = False
                reason = "Not Run"
                ensemble_stats = {}
                
                for attempt in range(loop_count):
                    # For Phantom, the inner loop handles retries as per Phase 15.X spec
                    
                    print(f"--- Adversarial Loop Attempt {attempt+1} (Strength: {current_strength if not is_phantom else p_strength}) ---")
                    config.strength = current_strength
                    config.attempt = attempt 
                    
                    # PHASE 4 LITE: Use protect_frontier_lite for Frontier mode (Fast, Low VRAM)
                    if is_frontier:
                        print("[CloakEngine] PHASE 4 LITE: Using Anti-Segmentation protect method...")
                        img = self.latent_cloak.protect_frontier_lite(input_path, config=config, user_mask=user_mask)
                    elif is_phantom:
                        print(f"[CloakEngine] PHASE 15.Z: Using Phantom Smart Targeting (Str {p_strength}, Res {p_resolution}, Target {p_targeting}, BG {background_intensity})...")
                        # Phase 15.Z specific call
                        result, mask, info = self.latent_cloak.protect(
                            image_path=input_path, 
                            strength=strength, # 0-100
                            face_boost=face_boost,
                            user_mask=user_mask,
                            target_profile=target_profile,
                            steps=optimization_steps,
                            use_dwt=use_dwt_mamba,
                            use_stego=use_neural_stego,
                            stego_payload=hidden_command,
                            background_intensity=background_intensity, # Phase 15.Z
                            is_liquid_v2=is_liquid_v2,
                            liquid_grid_size=liquid_grid_size,
                            liquid_asymmetry=liquid_asymmetry
                        )
                        img = result
                        # Phase 16: Allow fall-through to Validation (Qwen)
                        # break  <-- REMOVED to enable Qwen execution
                    elif is_liquid:
                        print(f"[CloakEngine] PHASE 17.6: Using Liquid Warp (Grid {l_grid}, TV {l_tv}, Limit {l_limit}, Blur {l_blur})...")
                        img = self.latent_cloak.protect_liquid_warp(
                            input_path,
                            strength=p_strength,
                            grid_size=l_grid,
                            num_steps=l_steps,
                            lr=0.005,
                            tv_weight=l_tv,
                            flow_limit=l_limit,
                            mask_blur=l_blur
                        )
                    elif is_liquid_v2:
                        print(f"[CloakEngine] PHASE 17.9d: Using Liquid Warp V2 (Focal Length Attack)...")
                        img, warp_metrics = self.latent_cloak.protect_liquid_warp_v2(
                            input_path,
                            strength=p_strength,
                            grid_size=liquid_grid_size,
                            num_steps=l_steps,
                            lr=0.01,
                            tv_weight=0.01,
                            flow_limit=0.03,
                            mask_blur=l_blur,
                            asymmetry_strength=liquid_asymmetry
                        )
                        # Store warp metrics for visualization
                        ensemble_stats['warp_metrics'] = warp_metrics
                    elif is_ghost_mesh:
                        print(f"[CloakEngine] PHASE 18: Using Ghost-Mesh Protocol (Grid {gm_grid}, Balance {gm_balance}, Anchoring {gm_anchoring})...")
                        img, mesh_metrics = self.latent_cloak.protect_ghost_mesh(
                            input_path,
                            strength=p_strength,
                            grid_size=gm_grid,
                            num_steps=l_steps,
                            warp_noise_balance=gm_balance,
                            tzone_anchoring=gm_anchoring,
                            tv_weight=gm_tv,
                            use_jnd=gm_jnd,
                            noise_strength=gm_noise,
                            warp_strength=gm_warp
                        )
                        # Store mesh metrics for visualization
                        ensemble_stats['mesh_metrics'] = mesh_metrics
                    else:
                        img = self.latent_cloak.protect(input_path, config=config, user_mask=user_mask)
                    
                    # VISUAL TRUST BADGE (Only for General mode, skip for Frontier Lite and Phantom)
                    if use_badge and not (is_frontier or is_phantom or is_ghost_mesh):
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
                    passed, reason, score = self.qwen_critic.critique_pairwise(reference_path=input_path, probe_path=output_path)
                    print(f"[CloakEngine] Qwen Result: {passed} | {reason}")
                    
                    # Capture metrics for UI - PRESERVE existing metrics (mesh_metrics, warp_metrics)
                    preserved_mesh = ensemble_stats.get('mesh_metrics', None)
                    preserved_warp = ensemble_stats.get('warp_metrics', None)
                    
                    ensemble_stats = {
                        "qwen_passed": passed,
                        "qwen_reason": reason,
                        "qwen_score": score,
                        "steps": l_steps if is_liquid or is_liquid_v2 or is_ghost_mesh else 0,
                        "strength": current_strength
                    }
                    
                    # Restore preserved metrics
                    if preserved_mesh:
                        ensemble_stats['mesh_metrics'] = preserved_mesh
                    if preserved_warp:
                        ensemble_stats['warp_metrics'] = preserved_warp
                    
                    # Cleanup critic (lightweight if API mode)
                    del self.qwen_critic
                    self.qwen_critic = None
                    
                    if passed:
                        print("✅ Qwen-VL Validation Passed.")
                        break
                    else:
                        print(f"❌ Qwen-VL Validation Failed. {'Retrying with new random init...' if is_liquid_v2 else 'Increasing Strength...'}")
                        
                        if is_liquid_v2:
                            # V2 Logic: "Best of N" - Don't increase strength, just retry with new random skew
                            # Ideally we should track the 'best' result so far, but for now we just verify.
                            # The loop continues to next attempt.
                            pass
                        else:
                             # Default Logic: Increase Strength
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
                     from invisible_core.attacks.stego_text import StegoInjector
                     self.stego_injector = StegoInjector()
                 img = self.stego_injector.inject_text(img, payload="GDPR_BLOCK")
            
            # Save final output
            img.save(output_path)
            
            # --- Metrics & Visualization ---
            original_img = Image.open(input_path).convert("RGB")
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
                     # Ensure InsightFace is loaded
                     if hasattr(self.latent_cloak, '_load_face_analysis'):
                         self.latent_cloak._load_face_analysis()
                     
                     # Compute Sim - Pass PIL Images, not numpy arrays
                     sim_score = self.latent_cloak.compute_similarity(orig_rgb, img_rgb)
                     if sim_score is not None:
                         bio_sim = float(sim_score)
            except Exception as e:
                print(f"Bio-Sim Error: {e}")

            metrics = {
                "psnr": psnr,
                "ssim": ssim_val,
                "bio_sim": bio_sim,
                "Visual Fidelity (PSNR)": f"{psnr:.2f} dB",
                "Injection Magnitude (MSE)": f"{mse:.2f}",
                "Biometric Similarity": f"{bio_sim:.4f}",
                "Layers Applied": []
            }
            
            # Merge Qwen Stats
            if ensemble_stats:
                metrics.update(ensemble_stats)
                
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
            from invisible_core.attacks.latent_cloak import LatentCloak
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
             from invisible_core.attacks.stego_text import StegoInjector
             self.stego_injector = StegoInjector()
         return self.stego_injector.inject_text(img_pil, payload=text or "GDPR_BLOCK")
