"""
Project Invisible - Streamlit Dashboard
AI Identity Defense Command Center
"""
import streamlit as st
import streamlit.elements.image as st_image
import streamlit.runtime as st_runtime
from io import BytesIO
import sys
from src.core.session_manager import LiveSessionManager
from src.views.live_dashboard import render_live_dashboard

@st.cache_resource
def get_live_manager():
    return LiveSessionManager()

@st.cache_resource
def get_qwen_critic():
    from src.core.critics.qwen_critic import QwenCritic
    return QwenCritic()

@st.cache_resource
def load_face_analysis():
    import insightface
    try:
        app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    except:
        app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

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
        
        # --- Liquid Warp V2 Controls ---
        # This toggle will override the target_profile_val for Liquid Warp V2 if enabled
        is_liquid_v2_toggle = st.toggle("Enable Liquid Warp V2 (Phase 17.9)", value=False, help="Use advanced anchored warping (Phase 17.9)")
        
        liquid_grid_size = 12 # Default (Phase 17.9d)
        liquid_asymmetry = 0.10 # Default
        
        if is_liquid_v2_toggle:
            st.markdown("### 💧 Liquid V2 Settings")
            liquid_grid_size = st.select_slider(
                "Grid Resolution",
                options=[12, 14, 16, 24, 32, 48, 64],
                value=12,
                help="Lower = smoother warps (12 recommended). Higher = finer distortions."
            )
            
            liquid_asymmetry = st.slider(
                "Asymmetry Intensity", 
                min_value=0.0, 
                max_value=0.3, 
                value=0.10, 
                step=0.01,
                help="Controls the strength of the 'Picasso' skew effect."
            )

        # Target Profile
        target_profile_options = {
            "Ghost-Mesh (Phase 18)": "ghost_mesh",
            "Liquid Warp V2 (Phase 17.9)": "liquid_17_v2",
            "Liquid Warp (Phase 17)": "liquid_17",
            "Resonant Ghost (Phase 16)": "phantom_15",
            "Frontier Lite (Anti-Grok)": "frontier",
            "General (SD/Midjourney)": "general"
        }
        
        # If Liquid Warp V2 toggle is on, force the selection
        if is_liquid_v2_toggle:
            selected_profile_label = "Liquid Warp V2 (Phase 17.9)"
            st.selectbox(
                "Target Profile",
                options=list(target_profile_options.keys()),
                index=list(target_profile_options.keys()).index(selected_profile_label),
                help="Liquid Warp V2: Anchored T-zone warp. Liquid Warp: Geometric warping. Resonant Ghost: Pixel perturbation.",
                disabled=True # Disable selection as it's forced by toggle
            )
        else:
            selected_profile_label = st.selectbox(
                "Target Profile",
                options=list(target_profile_options.keys()),
                help="Liquid Warp V2: Anchored T-zone warp. Liquid Warp: Geometric warping. Resonant Ghost: Pixel perturbation."
            )

        target_profile_val = target_profile_options[selected_profile_label]
        is_frontier = (target_profile_val == "frontier")
        is_phantom = (target_profile_val == "phantom_15")
        is_liquid = (target_profile_val == "liquid_17")
        # Determine is_liquid_v2 based on the toggle, otherwise from target_profile_val
        is_liquid_v2 = is_liquid_v2_toggle or (target_profile_val == "liquid_17_v2")
        is_ghost_mesh = (target_profile_val == "ghost_mesh")
        
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
        elif is_liquid_v2:
            st.info("🌊 **Phase 17.9 (Liquid Warp V2)**\n\nT-Zone Anchoring: Warps internal features while preserving silhouette.")
            
            with st.expander("🌊 Liquid Warp V2 Controls", expanded=True):
                # 1. Intensity (Strength)
                p_strength = st.slider(
                    "Warp Intensity", 
                    min_value=1, max_value=100, 
                    value=st.session_state.get('phantom_strength', 75), 
                    help="Higher = Stronger identity disruption."
                )
                
                # 2. Grid Resolution (now uses the `liquid_grid_size` from the toggle section)
                # This is already set by the toggle section, just display info or re-use
                st.caption(f"Warp Frequency (Grid): {liquid_grid_size}x{liquid_grid_size}")
                
                # 3. Optimization Steps
                optimization_steps = st.slider(
                    "Warp Cycles",
                    min_value=50, max_value=300, value=100, step=25,
                    help="More cycles = deeper identity erasure."
                )
                
                # 4. Mask Softness (Blur Kernel)
                mask_blur = st.slider(
                    "Mask Softness",
                    min_value=1, max_value=31, value=15, step=2,
                    help="Higher = Smoother T-zone transition."
                )
                
                # 5. Total Runs (Best of N)
                num_runs = st.slider(
                    "Total Runs (Best of N)",
                    min_value=1, max_value=5, value=1,
                    help="Run multiple times with different random skews and pick best result. 1 = Single run."
                )
                
                # Asymmetry Intensity (now uses the `liquid_asymmetry` from the toggle section)
                st.caption(f"Asymmetry Intensity: {liquid_asymmetry:.2f}")
                
                st.caption("⚙️ V2b Defaults: TV=0.01, FlowLimit=0.03")
                
                # Save to session state
                st.session_state['phantom_strength'] = p_strength
                st.session_state['liquid_grid'] = liquid_grid_size # Use the value from the toggle section
                st.session_state['liquid_tv'] = 0.01  # Fixed for V2
                st.session_state['liquid_steps'] = optimization_steps
                st.session_state['liquid_limit'] = 0.03  # Fixed for V2
                st.session_state['liquid_blur'] = mask_blur
                st.session_state['phantom_retries'] = num_runs  # Use retries var for runs
                st.session_state['liquid_asymmetry'] = liquid_asymmetry # Save asymmetry
            
            ensemble_diversity = 1
            use_dwt_mamba = False
            use_neural_stego = False
            hidden_command = ""
            anti_heal_intensity = 0.0
            max_retries_ui = num_runs # Correctly pass Total Runs
            p_targeting = 1.0
            p_background = 0.0
            p_resolution = "Original"
            p_retries = 1
            flow_limit = 0.03
            p_tv_weight = 0.01
        elif is_ghost_mesh:
            st.info("👻🕸️ **Phase 18 (Ghost-Mesh)**\n\nCoupled Warp + Noise Optimization with Hinge-Loss Constraints.")
            
            with st.expander("👻🕸️ Ghost-Mesh Controls", expanded=True):
                # Granular Controls Toggle
                granular_controls = st.toggle("Granular Controls", value=True, help="Separate Warp vs Noise strength.")
                
                if granular_controls:
                    gm_noise = st.slider("Noise Intensity", 0, 100, 75, help="Pixel perturbation strength.")
                    gm_warp = st.slider("Warp Intensity", 0, 100, 50, help="Geometric distortion strength.")
                    
                    st.session_state['ghost_mesh_noise'] = gm_noise
                    st.session_state['ghost_mesh_warp'] = gm_warp
                    
                    # Dummy values for legacy logic
                    p_strength = max(gm_noise, gm_warp)
                    gm_balance = 0.5
                else:
                    st.session_state['ghost_mesh_noise'] = None
                    st.session_state['ghost_mesh_warp'] = None
                    
                    # 1. Attack Intensity (Strength)
                    p_strength = st.slider(
                        "Attack Intensity", 
                        min_value=1, max_value=100, 
                        value=st.session_state.get('phantom_strength', 75), 
                        help="Higher = Stronger identity disruption."
                    )
                    
                    # 2. Warp/Noise Balance
                    gm_balance = st.slider(
                        "Warp/Noise Balance",
                        min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                        help="0.0 = Warp-heavy (geometric). 1.0 = Noise-heavy (pixel)."
                    )
                    st.session_state['ghost_mesh_balance'] = gm_balance
                
                # 3. Grid Resolution
                gm_grid_size = st.select_slider(
                    "Grid Resolution",
                    options=[12, 16, 24, 32],
                    value=24,
                    help="Low (12) = global shifts. High (32) = local distortion."
                )
                st.session_state['ghost_mesh_grid'] = gm_grid_size
                
                # 4. T-Zone Anchoring
                gm_anchoring = st.slider(
                    "T-Zone Anchoring",
                    min_value=0.0, max_value=1.0, value=0.8, step=0.1,
                    help="1.0 = Freeze silhouette/jawline. 0.0 = Full warp everywhere."
                )
                st.session_state['ghost_mesh_anchoring'] = gm_anchoring
                
                # 5. Grain Control (TV Weight)
                gm_tv = st.slider(
                    "Grain Control (TV)",
                    min_value=1, max_value=100, value=50,
                    help="Higher = Smoother noise. Lower = More grain."
                )
                st.session_state['ghost_mesh_tv'] = gm_tv
                
                # 6. Ghost Masking (JND)
                gm_jnd = st.checkbox(
                    "Ghost Masking (JND)", 
                    value=True,
                    help="Apply texture-only noise (invisible in smooth areas)."
                )
                st.session_state['ghost_mesh_jnd'] = gm_jnd
                
                # 7. Optimization Steps
                optimization_steps = st.slider(
                    "Optimization Cycles",
                    min_value=30, max_value=120, value=60, step=10,
                    help="More cycles = deeper identity erasure."
                )
                st.session_state['ghost_mesh_steps'] = optimization_steps
                
                # 8. Visualize Mesh Distortion (optional)
                st.divider()
                gm_show_mesh = st.toggle(
                    "Visualize Mesh Distortion", 
                    value=False, 
                    help="Show the geometric warp grid instead of difference heatmap."
                )
                st.session_state['gm_show_mesh'] = gm_show_mesh
                
                # Save to session state
                st.session_state['phantom_strength'] = p_strength
                st.session_state['ghost_mesh_grid'] = gm_grid_size
                st.session_state['ghost_mesh_balance'] = gm_balance
                st.session_state['ghost_mesh_anchoring'] = gm_anchoring
                st.session_state['ghost_mesh_tv'] = gm_tv
                st.session_state['ghost_mesh_jnd'] = gm_jnd
                st.session_state['liquid_steps'] = optimization_steps
            
            ensemble_diversity = 1
            use_dwt_mamba = False
            use_neural_stego = False
            hidden_command = ""
            anti_heal_intensity = 0.0
            max_retries_ui = st.session_state.get('phantom_retries', 3)
            p_targeting = 1.0
            p_background = 0.0
            p_resolution = "Original"
            p_retries = 1
            flow_limit = 0.03
            p_tv_weight = 0.01
            liquid_grid_size = gm_grid_size
            liquid_asymmetry = 0.10
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
                
                # Live Mode Redirect (Phase 18)
                if is_ghost_mesh:
                     live_params = {
                          'image_path': input_path,
                          'face_analysis': load_face_analysis(), # Inject Face Detection
                          'strength': p_strength, 
                          'grid_size': st.session_state.get('ghost_mesh_grid', 24),
                          'num_steps': optimization_steps,
                          'warp_noise_balance': st.session_state.get('ghost_mesh_balance', 0.5),
                          'tzone_anchoring': st.session_state.get('ghost_mesh_anchoring', 0.8),
                          'tv_weight': st.session_state.get('ghost_mesh_tv', 50),
                          'use_jnd': st.session_state.get('ghost_mesh_jnd', True),
                          'lr': 1.0, # Tanh params need high LR (0.05 * 20 = 1.0)
                          'noise_strength': st.session_state.get('ghost_mesh_noise'),
                          'warp_strength': st.session_state.get('ghost_mesh_warp')
                     }
                     
                     # Validation Callback
                     def validate_api(pil_img, ref_img=None):
                         import uuid, os
                         critic = get_qwen_critic()
                         
                         tmp_probe = f"tmp_prb_{uuid.uuid4().hex}.png"
                         tmp_ref = f"tmp_ref_{uuid.uuid4().hex}.png"
                         
                         pil_img.save(tmp_probe)
                         
                         try:
                             if ref_img:
                                 ref_img.save(tmp_ref)
                                 passed, reason, score = critic.critique_pairwise(tmp_ref, tmp_probe)
                             else:
                                 passed, reason, score = critic.critique(tmp_probe, "Target")
                             return reason
                         except Exception as e:
                             return f"Error: {str(e)}"
                         finally:
                             if os.path.exists(tmp_probe): os.remove(tmp_probe)
                             if os.path.exists(tmp_ref): os.remove(tmp_ref)
                             
                     live_manager = get_live_manager()
                     live_manager.init_session(engine.get_ghost_mesh_optimizer(), live_params, validator_fn=validate_api)
                     st.session_state['live_session_active'] = True
                     st.rerun()

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
                    background_intensity=p_background if is_phantom else 1.0,
                    # Phase 17.9c: Granular Control
                    is_liquid_v2=is_liquid_v2,
                    liquid_grid_size=liquid_grid_size,
                    liquid_asymmetry=liquid_asymmetry
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
                            <span style="color: #94a3b8;">[INFO]</span> Trust Badge: <span style="color: #4ade80;">INJECTED (Invisible)</span><br>
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
                        
                        # Qwen Analysis Display
                        qwen_passed = metrics.get('qwen_passed', None)
                        if qwen_passed is not None:
                            q_color = "#4ade80" if qwen_passed else "#ef4444"
                            q_icon = "✅" if qwen_passed else "❌"
                            q_score = metrics.get('qwen_score', 0.0) * 100
                            st.markdown(f"""
                            <div style="background-color: #1e1e24; border: 1px solid {q_color}; border-radius: 8px; padding: 12px; margin-top: 10px;">
                                <h4 style="color: {q_color}; margin: 0;">{q_icon} Qwen-VL Analysis</h4>
                                <p style="color: #a0a0a0; font-size: 0.9em; margin: 5px 0;">
                                    <strong>Match Score:</strong> {q_score:.1f}%<br>
                                    <strong>Verdict:</strong> {metrics.get('qwen_reason', 'Analysis complete.')}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Panic Mode Alert (Phase 18)
                            if not qwen_passed and is_ghost_mesh:
                                st.markdown("""
                                <div style="background-color: #7f1d1d; border: 2px solid #ef4444; border-radius: 8px; padding: 15px; margin-top: 10px; animation: pulse 1.5s infinite;">
                                    <h4 style="color: #fca5a5; margin: 0;">⚠️ PANIC MODE TRIGGERED</h4>
                                    <p style="color: #fecaca; font-size: 0.9em; margin: 5px 0;">
                                        Qwen-VL detected identity match. Retry loop initiated with randomized parameters.
                                        Consider increasing <strong>Attack Intensity</strong> or adjusting <strong>Warp/Noise Balance</strong>.
                                    </p>
                                </div>
                                <style>
                                @keyframes pulse {
                                    0%, 100% { opacity: 1; }
                                    50% { opacity: 0.7; }
                                }
                                </style>
                                """, unsafe_allow_html=True)
                        
                        # Visualization Graphs (Phase 17.9d)
                        warp_metrics = metrics.get('warp_metrics', None)
                        if warp_metrics and len(warp_metrics.get('step', [])) > 0:
                            import matplotlib.pyplot as plt
                            import matplotlib
                            matplotlib.use('Agg')  # Non-interactive backend
                            
                            st.markdown("### 📊 Optimization Visualization")
                            
                            # Create figure with two subplots
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor='#0e0e12')
                            
                            steps = warp_metrics['step']
                            avg_sim = warp_metrics['avg_sim']
                            tv_loss = warp_metrics['tv_loss']
                            disp_max = warp_metrics['disp_max']
                            
                            # Graph 1: AI Identity Convergence
                            ax1.set_facecolor('#0e0e12')
                            ax1.plot(steps, avg_sim, 'o-', color='#3b82f6', linewidth=2, markersize=4, label='Avg Similarity (AI)')
                            ax1.axhline(y=0.60, color='#ef4444', linestyle='--', linewidth=1.5, label='Target Breaking Threshold (0.60)')
                            ax1.set_xlabel('Step', color='white')
                            ax1.set_ylabel('Similarity Score', color='white')
                            ax1.set_title('AI Identity Convergence (AvgSim)', color='white', fontsize=12)
                            ax1.tick_params(colors='white')
                            ax1.legend(loc='upper right', facecolor='#1e1e24', edgecolor='white', labelcolor='white')
                            ax1.set_ylim(0.5, 1.05)
                            ax1.grid(True, alpha=0.3, color='gray')
                            for spine in ax1.spines.values():
                                spine.set_color('gray')
                            
                            # Graph 2: Warp Dynamics & Penalty
                            ax2.set_facecolor('#0e0e12')
                            ax2_twin = ax2.twinx()
                            
                            line1, = ax2.plot(steps, tv_loss, 's-', color='#22c55e', linewidth=2, markersize=4, label='TV (Smoothness Penalty)')
                            line2, = ax2_twin.plot(steps, disp_max, '^-', color='#a855f7', linewidth=2, markersize=4, label='DispMax (Warp Intensity)')
                            
                            ax2.set_xlabel('Step', color='white')
                            ax2.set_ylabel('TV Score', color='#22c55e')
                            ax2_twin.set_ylabel('Max Displacement', color='#a855f7')
                            ax2.set_title('Warp Dynamics & Penalty Plateauing', color='white', fontsize=12)
                            ax2.tick_params(axis='y', colors='#22c55e')
                            ax2_twin.tick_params(axis='y', colors='#a855f7')
                            ax2.tick_params(axis='x', colors='white')
                            ax2.grid(True, alpha=0.3, color='gray')
                            for spine in ax2.spines.values():
                                spine.set_color('gray')
                            for spine in ax2_twin.spines.values():
                                spine.set_color('gray')
                            
                            # Combined legend
                            lines = [line1, line2]
                            labels = [l.get_label() for l in lines]
                            ax2.legend(lines, labels, loc='upper left', facecolor='#1e1e24', edgecolor='white', labelcolor='white')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        
                        # Visualization Graphs (Phase 18: Ghost-Mesh)
                        mesh_metrics = metrics.get('mesh_metrics', None)
                        if mesh_metrics and len(mesh_metrics.get('step', [])) > 0:
                            import matplotlib.pyplot as plt
                            import matplotlib
                            matplotlib.use('Agg')
                            
                            st.markdown("### 📊 Ghost-Mesh Optimization Visualization")
                            
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor='#0e0e12')
                            
                            steps = mesh_metrics['step']
                            identity_loss = mesh_metrics['identity_loss']
                            # Updated: Use new lpips_raw and lpips_hinge keys
                            lpips_raw = mesh_metrics.get('lpips_raw', mesh_metrics.get('lpips_loss', []))
                            lpips_hinge = mesh_metrics.get('lpips_hinge', [0] * len(steps))
                            tv_loss = mesh_metrics['tv_loss']
                            disp_max = mesh_metrics['disp_max']
                            
                            # Graph 1: Identity Disruption + LPIPS
                            ax1.set_facecolor('#0e0e12')
                            ax1.plot(steps, identity_loss, 'o-', color='#3b82f6', linewidth=2, markersize=4, label='Identity Loss (CosSim)')
                            ax1.plot(steps, lpips_raw, 's-', color='#f97316', linewidth=2, markersize=4, label='Raw LPIPS (Perceptual)')
                            ax1.plot(steps, lpips_hinge, '^-', color='#facc15', linewidth=2, markersize=3, alpha=0.7, label='LPIPS Hinge (Optimizer)')
                            ax1.axhline(y=0.60, color='#ef4444', linestyle='--', linewidth=1.5, label='Identity Breaking Threshold')
                            ax1.axhline(y=0.05, color='#f97316', linestyle=':', linewidth=1.5, alpha=0.7, label='LPIPS Threshold (τ=0.05)')
                            ax1.set_xlabel('Step', color='white')
                            ax1.set_ylabel('Loss Value', color='white')
                            ax1.set_title('Ghost-Mesh: Identity + Visual Quality', color='white', fontsize=12)
                            ax1.tick_params(colors='white')
                            ax1.legend(loc='upper right', facecolor='#1e1e24', edgecolor='white', labelcolor='white', fontsize=8)
                            ax1.set_ylim(-0.2, 1.1)  # Allow negative identity (anti-correlation)
                            ax1.grid(True, alpha=0.3, color='gray')
                            for spine in ax1.spines.values():
                                spine.set_color('gray')
                            
                            # Graph 2: Coupled Dynamics
                            ax2.set_facecolor('#0e0e12')
                            ax2_twin = ax2.twinx()
                            
                            line1, = ax2.plot(steps, tv_loss, 's-', color='#22c55e', linewidth=2, markersize=4, label='TV (Noise Smoothness)')
                            line2, = ax2_twin.plot(steps, disp_max, '^-', color='#a855f7', linewidth=2, markersize=4, label='DispMax (Warp Intensity)')
                            
                            ax2.set_xlabel('Step', color='white')
                            ax2.set_ylabel('TV Score', color='#22c55e')
                            ax2_twin.set_ylabel('Max Displacement', color='#a855f7')
                            ax2.set_title('Coupled Warp + Noise Dynamics', color='white', fontsize=12)
                            ax2.tick_params(axis='y', colors='#22c55e')
                            ax2_twin.tick_params(axis='y', colors='#a855f7')
                            ax2.tick_params(axis='x', colors='white')
                            ax2.grid(True, alpha=0.3, color='gray')
                            for spine in ax2.spines.values():
                                spine.set_color('gray')
                            for spine in ax2_twin.spines.values():
                                spine.set_color('gray')
                            
                            lines = [line1, line2]
                            labels = [l.get_label() for l in lines]
                            ax2.legend(lines, labels, loc='upper left', facecolor='#1e1e24', edgecolor='white', labelcolor='white')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            # Additional: Mesh Distortion Grid Visualization (if toggle enabled)
                            if st.session_state.get('gm_show_mesh', False):
                                st.markdown("### 🕸️ Mesh Distortion Grid")
                                st.caption("Showing the geometric warp field applied to the image.")
                                
                                # Create a simple grid visualization
                                import numpy as np
                                fig_mesh, ax_mesh = plt.subplots(1, 1, figsize=(8, 8), facecolor='#0e0e12')
                                ax_mesh.set_facecolor('#0e0e12')
                                
                                # Draw grid lines
                                grid_size = 12  # Default
                                for i in range(grid_size + 1):
                                    ax_mesh.axhline(y=i / grid_size, color='#3b82f6', alpha=0.5, linewidth=1)
                                    ax_mesh.axvline(x=i / grid_size, color='#3b82f6', alpha=0.5, linewidth=1)
                                
                                # Overlay distortion arrows (simulated from disp_max)
                                final_disp = mesh_metrics['disp_max'][-1] if mesh_metrics['disp_max'] else 0.01
                                arrow_scale = final_disp * 10
                                
                                for i in range(1, grid_size):
                                    for j in range(1, grid_size):
                                        x, y = j / grid_size, i / grid_size
                                        # Simulated radial displacement (like focal length bias)
                                        dx = (x - 0.5) * arrow_scale * np.exp(-((x-0.5)**2 + (y-0.5)**2) * 8)
                                        dy = 0  # Vertical penalty keeps dy small
                                        ax_mesh.arrow(x, y, dx, dy, head_width=0.01, head_length=0.005, 
                                                     fc='#a855f7', ec='#a855f7', alpha=0.7)
                                
                                ax_mesh.set_xlim(0, 1)
                                ax_mesh.set_ylim(0, 1)
                                ax_mesh.set_aspect('equal')
                                ax_mesh.set_title(f'Warp Field (Max Disp: {final_disp:.6f})', color='white', fontsize=12)
                                ax_mesh.tick_params(colors='white')
                                for spine in ax_mesh.spines.values():
                                    spine.set_color('gray')
                                
                                st.pyplot(fig_mesh)
                                plt.close(fig_mesh)
                        
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
                     # Correctly use the returned heatmap path
                     "mask": None
                 })
                 
                 # Attempt to load returned heatmap or mask
                 # Priority: heatmap_path returned > debug_perceptual_mask.png
                 mask_to_load = None
                 if 'heatmap_path' in locals() and heatmap_path and os.path.exists(heatmap_path):
                     mask_to_load = heatmap_path
                 elif os.path.exists(os.path.join(os.path.dirname(input_path), "debug_perceptual_mask.png")):
                     mask_to_load = os.path.join(os.path.dirname(input_path), "debug_perceptual_mask.png")
                 
                 if mask_to_load:
                     try:
                        st.session_state.history[-1]["mask"] = Image.open(mask_to_load).copy()
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
# ==================== LIVE DASHBOARD (Phase 18) ====================
if st.session_state.get('live_session_active', False):
    live_manager = get_live_manager()
    render_live_dashboard(live_manager)

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
