"""
Project Invisible - Streamlit Dashboard
AI Identity Defense Command Center
"""
import streamlit as st
import streamlit.elements.image as st_image
import streamlit.runtime as st_runtime
from io import BytesIO
import sys

# --- AGGRESSIVE STREAMLIT COMPATIBILITY PATCH (Phase 15.Z fix) ---
def image_to_url_patch(image, width, clamp, channels, output_format, image_id):
    try:
        if hasattr(image, 'save'):
            buffered = BytesIO()
            image.save(buffered, format=output_format.upper() if output_format else "PNG")
            content = buffered.getvalue()
        else:
            content = image
        runtime = st_runtime.get_instance()
        mimetype = f"image/{output_format.lower()}" if output_format else "image/png"
        return runtime.media_file_mgr.add(content, mimetype, image_id)
    except Exception:
        return ""

# 1. Patch the module object directly
st_image.image_to_url = image_to_url_patch
print(f"[DEBUG] st_image.image_to_url patched: {hasattr(st_image, 'image_to_url')}")

# 2. Patch sys.modules to ensure any future imports get the patched version
sys.modules['streamlit.elements.image'].image_to_url = image_to_url_patch
print(f"[DEBUG] sys.modules['streamlit.elements.image'] patched: {hasattr(sys.modules['streamlit.elements.image'], 'image_to_url')}")

# 3. Patch the canvas module directly after it's imported (later in the file)

import os
import sys
import uuid
import json
import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.cloaking import CloakEngine

from streamlit_drawable_canvas import st_canvas
import streamlit_drawable_canvas
if hasattr(streamlit_drawable_canvas, 'st_image'):
    streamlit_drawable_canvas.st_image.image_to_url = image_to_url_patch
    print(f"[DEBUG] streamlit_drawable_canvas.st_image patched: {hasattr(streamlit_drawable_canvas.st_image, 'image_to_url')}")
elif hasattr(streamlit_drawable_canvas, 'image_to_url'):
    streamlit_drawable_canvas.image_to_url = image_to_url_patch
    print(f"[DEBUG] streamlit_drawable_canvas.image_to_url patched: {hasattr(streamlit_drawable_canvas, 'image_to_url')}")
else:
    print("[DEBUG] Could not find image_to_url reference in streamlit_drawable_canvas")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Project Invisible | AI Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State for History
if 'history' not in st.session_state:
    st.session_state.history = []

# ==================== CUSTOM CSS ====================
def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    # Try loading from local path (dev) or container path
    css_path = os.path.join(os.path.dirname(__file__), "assets/style.css")
    load_css(css_path)
except FileNotFoundError:
    st.warning("⚠️ CSS Theme file not found. Falling back to default.")

# ==================== INITIALIZE ====================
@st.cache_resource
def load_engine():
    """Load the CloakEngine once and cache it."""
    return CloakEngine()

engine = load_engine()

# Ensure uploads directory exists
upload_dir = os.path.join(os.getcwd(), "uploads")
os.makedirs(upload_dir, exist_ok=True)

# ==================== HEADER ====================
# ==================== HEADER ====================
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; padding: 1.5rem; background: rgba(0,0,0,0.3); border: 1px solid var(--border-glass); border-radius: 12px; margin-bottom: 2rem;">
    <div style="display: flex; align-items: center;">
        <div class="status-dot"></div>
        <div>
            <h1 style="margin: 0; font-size: 1.8rem; line-height: 1.2;">PROJECT INVISIBLE</h1>
            <p style="margin: 0; font-size: 0.8rem; color: var(--neon-primary); opacity: 0.8;">AI DEFENSE SYSTEM // ACTIVE</p>
        </div>
    </div>
    <div style="text-align: right; border-left: 1px solid var(--border-glass); padding-left: 1.5rem;">
        <span style="font-family: 'JetBrains Mono'; color: #94a3b8; font-size: 0.75rem;">SECURE_CONNECTION: ENCRYPTED</span><br>
        <span style="font-family: 'JetBrains Mono'; color: var(--neon-secondary); font-size: 0.75rem;">VERSION 2.1.0 :: FRONTIER PROTOCOL</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ==================== SIDEBAR - CONTROLS ====================
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    
    # Defense Mode
    st.markdown("### 🛡️ Defense Mode")
    defense_mode = st.radio(
        "Protection Type",
        ["Full Defense", "AI Protection Only", "Badge Only"],
        index=0,
        label_visibility="collapsed"
    )
    use_ai = "AI" in defense_mode or "Full" in defense_mode
    use_badge = "Badge" in defense_mode or "Full" in defense_mode
    
    st.divider()
    
    # Advanced Operations (Method-specific controls inside)

    # 2. Advanced Frontier Settings (Hidden by default)
    with st.sidebar.expander("🛠️ Advanced Operations", expanded=False):
        st.markdown("### 🚀 Frontier Protocol")
        
        # Target Profile
        target_profile_options = {
            "Liquid Warp (Phase 17)": "liquid_17",
            "Resonant Ghost (Phase 16)": "phantom_15",
            "Frontier Lite (Anti-Grok)": "frontier",
            "General (SD/Midjourney)": "general"
        }
        selected_profile_label = st.selectbox(
            "Target Profile",
            options=list(target_profile_options.keys()),
            help="Liquid Warp: Geometric warping (invisible). Resonant Ghost: Pixel perturbation."
        )
        target_profile_val = target_profile_options[selected_profile_label]
        is_frontier = (target_profile_val == "frontier")
        is_phantom = (target_profile_val == "phantom_15")
        is_liquid = (target_profile_val == "liquid_17")
        
        # Phase 15.3: Resource Management
        if st.button("♻️ Flush VRAM (Release Memory)", help="Unload heavy AI models to free up GPU memory."):
            if hasattr(engine, 'latent_cloak') and engine.latent_cloak:
                if hasattr(engine.latent_cloak, 'unload_resources'):
                    engine.latent_cloak.unload_resources()
                    st.success("Memory Flushed.")
            else:
                st.info("Nothing to flush.")
        
        # Phase 4 Lite & Phase 15: Show different controls based on profile
        if is_frontier:
            st.info("⚡ **Phase 4 Lite Mode**\n\nFast anti-segmentation (~20 sec, <1GB VRAM)")
            
            # Anti-Heal Intensity (geometric warping)
            anti_heal_intensity = st.slider(
                "Geometric Warp",
                min_value=0.0, max_value=0.5, value=0.25, step=0.05,
                help="Creates wavy distortion to break AI segmentation. Higher = more visible warp."
            )
            
            # Fixed values for Phase 4 Lite (display info only)
            st.caption("⚙️ Auto-config: 20 steps, ε=0.05")
            
            # Disable legacy options for Frontier
            optimization_steps = 20  # Fixed
            ensemble_diversity = 1   # Not used
            use_dwt_mamba = False    # Not used
            use_neural_stego = False # Not used
            hidden_command = ""
        elif is_liquid:
            st.info("🌊 **Phase 17 (Liquid Warp)**\n\nGeometric warping. Invisible to humans, breaks AI identity.")
            
            with st.expander("🌊 Liquid Warp Controls", expanded=True):
                # 1. Intensity (Max Displacement Magnitude)
                p_strength = st.slider(
                    "Warp Intensity", 
                    min_value=1, max_value=100, 
                    value=st.session_state.get('phantom_strength', 75), 
                    help="Higher = Stronger identity disruption."
                )
                
                # 2. Grid Resolution (Default: Low = Safe)
                grid_option = st.selectbox(
                    "Warp Frequency (Grid)",
                    options=["Low (16x16)", "Medium (32x32)", "High (64x64)"],
                    index=0,  # Default to Low (16x16) for safety
                    help="Low = Safer (moves face chunks). High = May cause ripples."
                )
                p_grid_size = 16 if "Low" in grid_option else (64 if "High" in grid_option else 32)
                
                # 3. Coherence (TV Weight: 0-100 maps to 0-100)
                # Higher = Thicker liquid (more coherent movement)
                coherence = st.slider(
                    "Liquid Coherence (TV)",
                    min_value=0, max_value=100, value=50,  # Default 50 = TV weight 50
                    help="Higher = Thicker liquid (prevents tearing). Lower = Chaotic."
                )
                p_tv_weight = float(coherence)  # Direct mapping
                
                # 4. Flow Limit (Tanh Constraint - prevents melting)
                flow_limit = st.slider(
                    "Flow Limit (Max Shift)",
                    min_value=0.001, max_value=0.010, value=0.004, step=0.001,
                    format="%.3f",
                    help="Hard limit on pixel movement. Lower = Safer (less distortion)."
                )
                
                # 5. Mask Softness (Blur Kernel)
                mask_blur = st.slider(
                    "Mask Softness",
                    min_value=1, max_value=31, value=15, step=2,
                    help="Higher = Smoother face/background transition. Prevents 'cut-out' look."
                )
                
                # 6. Optimization Steps
                optimization_steps = st.slider(
                    "Warp Cycles",
                    min_value=50, max_value=300, value=100, step=25,
                    help="More cycles = deeper identity erasure."
                )

                # 7. Warp Attempts (Retry with different random init)
                p_retries = st.slider(
                    "Warp Attempts",
                    min_value=1, max_value=5, value=1,
                    help="Multiple attempts with different initializations."
                )

                # Save to session state
                st.session_state['phantom_strength'] = p_strength
                st.session_state['liquid_grid'] = p_grid_size
                st.session_state['liquid_tv'] = p_tv_weight
                st.session_state['liquid_steps'] = optimization_steps
                st.session_state['liquid_limit'] = flow_limit
                st.session_state['liquid_blur'] = mask_blur
                st.session_state['phantom_retries'] = p_retries
            
            ensemble_diversity = 1
            use_dwt_mamba = False
            use_neural_stego = False
            hidden_command = ""
            anti_heal_intensity = 0.0
            max_retries_ui = p_retries
            p_targeting = 1.0 # Default face focus
            p_background = 0.0 
            p_resolution = "Original"
        elif is_phantom:
            st.info("👻 **Resonant Ghost (Phase 16)**\n\nSigLIP + MI-FGSM. Pixel perturbation attack.")
            
            with st.expander("👻 Phantom Controls (Phase 15.X/Y)", expanded=True):
                # 1. Strength Slider (Gas Pedal)
                p_strength = st.slider(
                    "Attack Strength", 
                    min_value=1, 
                    max_value=100, 
                    value=st.session_state.get('phantom_strength', 50), 
                    help="Higher = Stronger attack. Strength > 80 may create visible grain."
                )
                
                # 2. Precision Targeting (Phase 15.Y/Z)
                p_targeting = st.slider(
                    "Targeting Intensity",
                    min_value=1.0,
                    max_value=10.0,
                    value=st.session_state.get('phantom_targeting', 3.0),
                    step=0.5,
                    help="Multiplier for the 'Power Zone' painted with the brush tool."
                )

                # NEW: Background Intensity (Phase 15.Z)
                p_background = st.slider(
                    "Background Intensity",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get('phantom_background', 0.2),
                    step=0.05,
                    help="Noise level for non-targeted areas. 0.0 = Pure Stealth."
                )

                # 3. Attack Resolution (Phase 15.Y)
                p_resolution = st.selectbox(
                    "Attack Resolution",
                    options=[224, 384, 512, 768, "Original"],
                    index=1, # Default 384
                    help="Optimizing at lower resolution is faster and uses less VRAM."
                )
                
                # 4. Retry Limit (Persistence)
                p_retries = st.number_input(
                    "Max Retries", 
                    min_value=1, 
                    max_value=10, 
                    value=st.session_state.get('phantom_retries', 3),
                    help="If the attack fails, the AI will try again with new random noise."
                )
                
                # Save to session state
                st.session_state['phantom_strength'] = p_strength
                st.session_state['phantom_retries'] = p_retries
                st.session_state['phantom_targeting'] = p_targeting
                st.session_state['phantom_background'] = p_background
                st.session_state['phantom_resolution'] = p_resolution

            # Overrides for engine call
            optimization_steps = 40 # Base steps, but logic is handled in protect_phantom
            ensemble_diversity = 1
            use_dwt_mamba = False
            use_neural_stego = False
            hidden_command = ""
            anti_heal_intensity = 0.0
            max_retries_ui = p_retries # Sync with the new slider
        else:
            # Legacy General mode controls
            max_retries_ui = 5 # Default for other modes
            # Legacy General mode controls
            optimization_steps = st.slider(
                "Optimization Cycles",
                min_value=50, max_value=500, value=100, step=25
            )
            
            ensemble_diversity = st.slider(
                "Ensemble Diversity",
                min_value=1, max_value=10, value=5
            )
            
            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                use_dwt_mamba = st.checkbox("DWT-Mamba", value=True)
            with col_adv2:
                use_neural_stego = st.checkbox("Neural Stego", value=True)
                
            hidden_command = ""
            if use_neural_stego:
                hidden_command = st.text_input("Payload", value="IGNORE_IDENTITY")
            
            anti_heal_intensity = 0.25  # Default for general mode

    # 3. Targeting & Compliance
    st.sidebar.markdown("### 🎯 Targeting & Compliance")
    
    face_boost = st.sidebar.checkbox(
        "👤 Face Boost (4x)", value=True, help=" prioritize facial identity protection"
    )
    
    compliance_layer = st.sidebar.checkbox(
        "📝 GDPR Watermark", value=True
    )
    
    # Badge (Always on by default for trust)
    use_badge = True
    badge_text = "PROTECTED" # Default
    
    # Visual mode is always diffusion for this pro app
    visual_mode = "latent_diffusion"
    
    # draw_mode is removed from sidebar, set to True by default
    # Phase 15.Y/Z: Always enable draw mode for Phantom to allow Precision Targeting
    draw_mode = True if is_phantom else False


# ==================== MAIN CONTENT ====================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📡 UPLINK_FEED (SOURCE)")
    st.caption("Inject biometric data for shielding.")
    
    uploaded_file = st.file_uploader(
        "Upload your image",
        type=["png", "jpg", "jpeg", "webp"],
        help="Supported formats: PNG, JPG, JPEG, WEBP"
    )
    
    # Canvas for drawing (if draw mode enabled)
    user_mask = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        # --- PHOTO INTELLIGENCE (Phase 15.Z) ---
        # Gating: Only show for PHANTOM (Phase 15+)
        if is_phantom:
            with st.expander("🔍 PHOTO INTELLIGENCE", expanded=True):
                w_orig, h_orig = image.size
                st.markdown(f"**Resolution:** `{w_orig}x{h_orig}` | **Format:** `{uploaded_file.type}` | **Mode:** `{image.mode}`")
                
                col_i1, col_i2 = st.columns(2)
                with col_i1:
                    if st.button("✨ Auto-Target Faces"):
                        # Save temp for detection
                        temp_path = f"uploads/temp_{uuid.uuid4()}.png"
                        image.save(temp_path)
                        detections = engine.detect_faces(temp_path)
                        if detections:
                            st.session_state['auto_target_data'] = detections
                            st.success(f"Detected {len(detections)} targets!")
                        else:
                            st.warning("No targets detected.")
                        if os.path.exists(temp_path): os.remove(temp_path)
                with col_i2:
                    if st.button("🗑️ Clear Mask"):
                        st.session_state.pop('auto_target_data', None)
                        st.session_state.pop('auto_target_boxes', None) # Legacy cleanup
                        st.session_state['canvas_key'] = str(uuid.uuid4()) # Reset canvas
                        st.rerun()

        if draw_mode:
            st.markdown("*Precision Targeting: Draw on eyes, face, or sensitive areas for boosted protection.*")
            
            # Calculate canvas size (max 512px width)
            w, h = image.size
            scale = min(512 / w, 512 / h, 1.0)
            canvas_w, canvas_h = int(w * scale), int(h * scale)
            
            # Prepare initial drawing if auto-targeted (Phase 15.2 Smart Regions)
            initial_drawing = None
            if is_phantom and 'auto_target_data' in st.session_state:
                # Convert landmarks to fabric.js polygons
                initial_drawing = {"objects": []}
                for det in st.session_state['auto_target_data']:
                    # New Format: dict with 'bbox' and optional 'kps'
                    # Or Legacy: list (bbox)
                    
                    if isinstance(det, dict) and 'kps' in det and det['kps'] is not None:
                        # Draw Smart Region (Polygon of landmarks)
                        kps = det['kps'] # 5 points: [L_Eye, R_Eye, Nose, L_Mouth, R_Mouth]
                        
                        # Phase 15.2: Create a Pentagon enclosing the features
                        # Scale points
                        points = [{"x": int(pt[0] * scale), "y": int(pt[1] * scale)} for pt in kps]
                        
                        initial_drawing["objects"].append({
                            "type": "polygon",
                            "points": points,
                            "fill": "rgba(255, 0, 0, 0.3)",
                            "stroke": "#FF0000",
                            "strokeWidth": 2,
                            "left": int(min(pt[0] for pt in kps) * scale),
                            "top": int(min(pt[1] for pt in kps) * scale)
                        })
                    else:
                        # Fallback to Bounding Box
                        box = det['bbox'] if isinstance(det, dict) else det
                        x1, y1, x2, y2 = box
                        initial_drawing["objects"].append({
                            "type": "rect",
                            "left": int(x1 * scale),
                            "top": int(y1 * scale),
                            "width": int((x2 - x1) * scale),
                            "height": int((y2 - y1) * scale),
                            "fill": "rgba(255, 0, 0, 0.3)",
                            "stroke": "#FF0000",
                            "strokeWidth": 2
                        })

            # FINAL BRUTE FORCE PATCH (Phase 15.Z Streamlit 1.53 fix)
            try:
                import streamlit.elements.image as st_img
                st_img.image_to_url = image_to_url_patch
                import streamlit_drawable_canvas
                if hasattr(streamlit_drawable_canvas, 'st_image'):
                    streamlit_drawable_canvas.st_image.image_to_url = image_to_url_patch
            except Exception:
                pass

            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=30,
                stroke_color="#FF0000",
                background_image=image.resize((canvas_w, canvas_h)),
                initial_drawing=initial_drawing,
                update_streamlit=True,
                height=canvas_h,
                width=canvas_w,
                drawing_mode="freedraw" if initial_drawing is None else "transform",
                key=st.session_state.get('canvas_key', "canvas")
            )
            
            # Extract mask from canvas
            if canvas_result.image_data is not None:
                mask_rgba = canvas_result.image_data
                if mask_rgba.shape[-1] == 4:
                    user_mask = (mask_rgba[:, :, 3] > 0).astype(np.float32)
        else:
            st.image(image, caption="Original Asset", width="content")

with col2:
    st.markdown("### 🖥️ SYSTEM_TERMINAL")
    
    # Placeholder for output
    terminal_placeholder = st.empty()
    output_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    # In draw_mode, hidden drawing canvas logic is simpler if we assume None
    if not draw_mode:
        pass

# ==================== PROCESS BUTTON ====================
if st.button("⚡ ACTIVATE DEFENSE", type="primary"):
    if uploaded_file is None:
        st.error("❌ Please upload an image first.")
    else:
        spinner_text = "🔄 INITIALIZING DEFENSE PROTOCOLS..."
        if is_frontier:
            spinner_text = "⚡ PHASE 4 LITE: Anti-Segmentation Mode (Fast, <1GB VRAM)"
        elif is_phantom:
            spinner_text = "👻 PHASE 15.5: Phantom Pixel Boost (SigLIP + JND Heatmap)"
        
        with st.spinner(spinner_text):
            try:
                # Save input image
                input_filename = f"{uuid.uuid4()}_input.png"
                input_path = os.path.join(upload_dir, input_filename)
                image.save(input_path)
                
                output_filename = f"cloaked_{uuid.uuid4().hex}.png"
                output_path = os.path.join(upload_dir, output_filename)
                
                # Determine face boost value
                face_boost_val = 2.5 if face_boost else 1.0
                
                # Apply Defense with Mode Logic
                visual_mode_arg = "latent_diffusion" if use_ai else "none"
                
                # Determine target profile for engine
                # target_profile_val is already set from sidebar
                
                success, heatmap_path, metrics = engine.apply_defense(
                    input_path,
                    output_path,
                    visual_mode=visual_mode,
                    compliance=compliance_layer,
                    strength=p_strength,
                    face_boost=face_boost,
                    user_mask=user_mask,
                    use_badge=use_badge,
                    badge_text=badge_text,
                    # Phase 2 options
                    target_profile=target_profile_val,
                    ensemble_diversity=ensemble_diversity,
                    optimization_steps=optimization_steps,
                    use_dwt_mamba=use_dwt_mamba,
                    use_neural_stego=use_neural_stego,
                    hidden_command=hidden_command,
                    max_retries=max_retries_ui,
                    background_intensity=p_background if is_phantom else 1.0
                )
                
                if success:
                    # Display results
                    with col2:
                        # Terminal Log
                        # Determine actual params used (Phase 5.2 Override)
                        actual_steps = 25 if is_frontier else (40 if is_phantom else optimization_steps)
                        actual_dwt = "ACTIVE" if (is_frontier or use_dwt_mamba) else "Inactive"
                        
                        terminal_placeholder.markdown(f"""
                        <div class="terminal-container">
                            <span style="color: #4ade80;">[SUCCESS]</span> Defense Matrix Applied.<br>
                            <span style="color: #94a3b8;">[INFO]</span> Optimization Steps: {actual_steps}<br>
                            <span style="color: #94a3b8;">[INFO]</span> DWT-Mamba Loss: {actual_dwt}<br>
                            <span style="color: #94a3b8;">[INFO]</span> Neural Payload: {hidden_command if use_neural_stego else 'NONE'}<br>
                            <span style="color: #facc15;">[WARN]</span> Bio-Signature Scrambled.<br>
                            <span style="color: #4ade80;">[SYSTEM]</span> Output generated at {output_path}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        output_placeholder.image(output_path, caption="🔒 SECURE_OUTPUT_V2.png", width="stretch")
                        
                        # Custom Metrics Grid
                        if metrics:
                            psnr_val = metrics.get('psnr', 0.0)
                            ssim_val = metrics.get('ssim', 0.0)
                            bio_val = metrics.get('bio_sim', 0.0)
                            
                            metrics_placeholder.markdown(f"""
                            <div class="metric-grid">
                                <div class="metric-box">
                                    <div class="metric-label">Signal Clarity (PSNR)</div>
                                    <div class="metric-value">{psnr_val:.1f} dB</div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-label">Structure (SSIM)</div>
                                    <div class="metric-value">{ssim_val:.3f}</div>
                                </div>
                                <div class="metric-box" style="border-color: {'#4ade80' if bio_val < 0.4 else '#ef4444'};">
                                    <div class="metric-label">Bio-Signature</div>
                                    <div class="metric-value" style="color: {'#4ade80' if bio_val < 0.4 else '#ef4444'};">{bio_val:.3f}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Download button
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="💾 EXPORT SECURE ASSET",
                                data=f,
                                file_name=f"secure_{output_filename}",
                                mime="image/png",
                                type="secondary"
                            )
                    
                    frontier_msg = ""
                    if is_frontier:
                        frontier_msg = f"""
                        🚀 <strong>FRONTIER MODE</strong><br>
                        Ensemble Diversity: {ensemble_diversity}/10<br>
                        Steps: {optimization_steps}<br>
                        {'🌊 DWT-Mamba: ON' if use_dwt_mamba else ''}<br>
                        {'🧠 Neural Stego: ON' if use_neural_stego else ''}
                        """
                    elif is_phantom:
                        frontier_msg = f"""
                        👻 <strong>PHANTOM PIXEL</strong><br>
                        SigLIP + JND Stealth<br>
                        Steps: 40 (Precision)<br>
                        """
                    
                    st.markdown(f"""
                    <div class="status-success">
                        ✅ <strong>DEFENSE ACTIVATED</strong><br>
                        🎯 Intensity: {p_strength}%<br>
                        {'🔥 MAXIMUM' if p_strength >= 80 else '⚡ Standard' if p_strength >= 50 else '👻 Stealth'}<br>
                        {'🎨 Targeted Areas Boosted' if user_mask is not None else ''}
                        {'👤 Face Priority: ON' if face_boost else ''}
                        {frontier_msg}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("❌ Defense execution failed. Check logs.")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        # Add to history if successful
        if 'output_path' in locals() and 'input_path' in locals() and os.path.exists(output_path) and metrics:
             # Load images from disk paths (not in-memory variables)
             try:
                 original_for_history = Image.open(input_path).copy()
                 protected_for_history = Image.open(output_path).copy()
                 
                 st.session_state.history.append({
                     "original": original_for_history,
                     "protected": protected_for_history,
                     "metrics": metrics,
                     "timestamp": uuid.uuid4().hex[:8],
                     # Try to find and load the perceptual mask
                     "mask": None
                 })
                 
                 # Attempt to load Phase 10 mask
                 mask_path = os.path.join(os.path.dirname(input_path), "debug_perceptual_mask.png")
                 if os.path.exists(mask_path):
                     try:
                        st.session_state.history[-1]["mask"] = Image.open(mask_path).copy()
                     except:
                        pass
                 if len(st.session_state.history) > 5:
                     st.session_state.history.pop(0)
             except Exception as hist_err:
                 print(f"[History] Failed to save: {hist_err}")

# ==================== DEFENSE HISTORY ====================
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 Defense History (Session)")
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Run #{len(st.session_state.history)-i} - Bio-Sim: {item['metrics'].get('bio_sim', 'N/A')}", expanded=False):
            hc1, hc2, hc3 = st.columns([1, 1, 1])
            with hc1:
                st.image(item['original'], caption="Original", width="stretch")
            with hc2:
                st.image(item['protected'], caption="Protected", width="stretch")
            with hc3:
                st.json(item['metrics'])
# ==================== FOOTER ====================
st.divider()

# ==================== AI VISION REPORT (Phase 4.7) ====================
st.markdown("### 👁️ AI Vision Report")
st.caption("Compare how AI sees your image before and after protection.")

if st.session_state.history:
    # Use most recent result for visualization
    latest = st.session_state.history[-1]
    
    with st.expander("🔬 View AI Segmentation Comparison", expanded=False):
        try:
            import cv2
            import numpy as np
            from ultralytics import YOLO
            
            @st.cache_resource
            def load_yolo():
                return YOLO("yolov8n-seg.pt")
            
            model = load_yolo()
            
            # Get images from session
            img_orig = np.array(latest['original'])
            img_prot = np.array(latest['protected'])
            
            # Resize protected to match original
            img_prot = cv2.resize(img_prot, (img_orig.shape[1], img_orig.shape[0]))
            
            # Run YOLO inference
            with st.spinner("Running AI segmentation analysis..."):
                res_orig = model(img_orig, verbose=False)[0]
                res_prot = model(img_prot, verbose=False)[0]
            
            # Extract person masks
            def get_person_conf(res):
                max_conf = 0.0
                for idx, cls in enumerate(res.boxes.cls.cpu().numpy()):
                    if int(cls) == 0:  # Person class
                        max_conf = max(max_conf, res.boxes.conf[idx].cpu().item())
                return max_conf
            
            def draw_boxes(img, res):
                img_draw = img.copy()
                for idx, cls in enumerate(res.boxes.cls.cpu().numpy()):
                    if int(cls) == 0:  # Person
                        box = res.boxes.xyxy[idx].cpu().numpy().astype(int)
                        conf = res.boxes.conf[idx].cpu().item()
                        cv2.rectangle(img_draw, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)
                        cv2.putText(img_draw, f"Person {conf:.2f}", (box[0], box[1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                return img_draw
            
            orig_conf = get_person_conf(res_orig)
            prot_conf = get_person_conf(res_prot)
            
            # Display comparison
            col_ai1, col_ai2, col_ai3, col_ai4 = st.columns([1, 1, 1, 1])
            
            with col_ai1:
                st.image(draw_boxes(img_orig, res_orig), caption=f"BEFORE: AI Detection ({orig_conf:.2f})", width="stretch")
            
            with col_ai2:
                st.image(draw_boxes(img_prot, res_prot), caption=f"AFTER: AI Vision ({prot_conf:.2f})", width="stretch")
            
            with col_ai3:
                # Perceptual Mask (Phase 10) or JND Heatmap (Phase 15.5)
                # Check for JND heatmap first if it was a phantom run
                jnd_path = "uploads/debug_jnd_heatmap.png"
                if is_phantom and os.path.exists(jnd_path):
                    st.image(jnd_path, caption="Phase 15.5 JND Heatmap", width="content")
                elif latest.get("mask") is not None:
                    st.image(latest["mask"], caption="Phase 10 Texture Mask", width="content")
                else:
                    st.info("No mask data")

            with col_ai4:
                # Difference map
                diff = cv2.absdiff(img_orig, img_prot)
                diff_amp = np.clip(diff * 20, 0, 255).astype(np.uint8)
                st.image(diff_amp, caption="Perturbation Map (x20)", width="stretch")
            
            # Status
            if prot_conf < 0.1:
                st.success(f"✅ **SUCCESS**: AI is BLIND to person (Conf: {orig_conf:.2f} → {prot_conf:.2f})")
            elif prot_conf < 0.5:
                st.warning(f"⚠️ **PARTIAL**: AI detection reduced (Conf: {orig_conf:.2f} → {prot_conf:.2f})")
            else:
                st.error(f"❌ **FAILED**: AI still sees person (Conf: {orig_conf:.2f} → {prot_conf:.2f})")
                
        except Exception as e:
            st.warning(f"AI Vision analysis unavailable: {e}")
else:
    st.info("Process an image to see the AI Vision comparison.")

st.divider()
st.markdown("""
<div style="text-align: center; opacity: 0.5; font-size: 0.85rem;">
    <strong>Defense Stack:</strong> YOLOv8n-Seg (Phase 4.7) • Raw Logit Attack<br>
    <em>Project Invisible © 2026 | Protecting Digital Identity</em>
</div>
""", unsafe_allow_html=True)
