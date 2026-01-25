import gradio as gr
import os
import sys
import uuid
import time
import shutil
import json
import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.cloaking import CloakEngine
from src.core.logger import logger, setup_logger

# Initialize Logger
setup_logger()

# Initialize engines
engine = CloakEngine()



# Ensure uploads directory exists
upload_dir = os.path.join(os.getcwd(), "uploads")
os.makedirs(upload_dir, exist_ok=True)

# ==================== CUSTOM CSS FOR PROFESSIONAL VISUALS ====================
custom_css = """
/* Professional Dark Theme */
.gradio-container {
    background-color: #0f172a !important;
}

/* Headers */
h1, h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

/* Buttons */
.primary {
    background-color: #3b82f6 !important;
    border-radius: 6px !important;
    transition: background 0.2s ease !important;
}
.primary:hover {
    background-color: #2563eb !important;
}

/* Cards/Containers */
.gr-group, .gr-box, .gr-panel {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}

/* Inputs */
input[type="range"] {
    accent-color: #3b82f6 !important;
}

/* Status Panel */
.status-box textarea {
    font-family: 'Consolas', 'Monaco', monospace !important;
    background-color: #0f172a !important;
    color: #4ade80 !important;
}
"""

# ==================== PROCESSING FUNCTIONS ====================

def extract_mask_from_editor(editor_data):
    """Extract binary mask from ImageEditor layers."""
    if editor_data is None:
        return None
    
    # ImageEditor returns dict with 'background', 'layers', 'composite'
    if isinstance(editor_data, dict):
        layers = editor_data.get("layers", [])
        if layers and len(layers) > 0:
            # Get the first layer (user drawings)
            layer = layers[0]
            if isinstance(layer, np.ndarray) and layer.shape[-1] == 4:
                # RGBA - use alpha channel as mask
                mask = layer[:, :, 3] > 0
                return mask.astype(np.float32)
            elif isinstance(layer, Image.Image):
                layer_np = np.array(layer)
                if layer_np.shape[-1] == 4:
                    mask = layer_np[:, :, 3] > 0
                    return mask.astype(np.float32)
    return None


def process_image_advanced(editor_data, destruction_level, face_boost, targeting_mode, compliance_layer, use_badge, badge_text):
    """
    Process the input image using Project Invisible with advanced targeting.
    """
    if editor_data is None:
        return None, None, None, None, "❌ Please upload an image first."
    
    print(f"[DEBUG] Processing Image. Type: {type(editor_data)}", flush=True)
    
    try:
        input_source = None
        
        # Extract background image path
        if isinstance(editor_data, dict):
            # Check for standard Gradio keys
            if "background" in editor_data:
                input_source = editor_data["background"]
            
            # If background is None or empty, try composite
            if input_source is None and "composite" in editor_data:
                input_source = editor_data["composite"]
                
        elif hasattr(editor_data, 'get'): # Duck typing check
             input_source = editor_data.get("background") or editor_data.get("composite")
        else:
             # Assume direct image/array
             input_source = editor_data
        
        # Proper None check for numpy arrays
        if input_source is None:
            print("[DEBUG] Input source is None after extraction.", flush=True)
            return None, None, None, None, "❌ No image found. Please upload an image."
            
        if isinstance(input_source, np.ndarray) and input_source.size == 0:
            print("[DEBUG] Input source is empty array.", flush=True)
            return None, None, None, None, "❌ Empty image. Please upload a valid image."
        
        # Save input to permanent path
        input_filename = f"{uuid.uuid4()}_input.png"
        input_path = os.path.join(upload_dir, input_filename)
        
        if isinstance(input_source, str):
            shutil.copy2(input_source, input_path)
        elif isinstance(input_source, np.ndarray):
            # Ensure proper format for saving
            if input_source.dtype != np.uint8:
                if input_source.max() <= 1.0:
                    input_source = (input_source * 255).astype(np.uint8)
                else:
                    input_source = input_source.astype(np.uint8)
            Image.fromarray(input_source).save(input_path)
        elif isinstance(input_source, Image.Image):
            input_source.save(input_path)
        else:
            return None, None, None, None, f"❌ Unsupported image type: {type(input_source)}"
        
        output_filename = f"cloaked_{uuid.uuid4().hex}.png"
        output_path = os.path.join(upload_dir, output_filename)
        
        # Extract user-drawn mask if targeting mode is enabled
        user_mask = None
        if targeting_mode:
            user_mask = extract_mask_from_editor(editor_data)
        
        # Map destruction level to face_boost value
        face_boost_val = 3.0 if face_boost else 1.0
        
        # Apply Defense with advanced parameters
        success, heatmap_path, metrics = engine.apply_defense(
            input_path, 
            output_path, 
            visual_mode="latent_diffusion", 
            compliance=compliance_layer,
            strength=destruction_level,
            face_boost=face_boost_val,
            user_mask=user_mask,
            use_badge=use_badge,
            badge_text=badge_text
        )
        
        if success:
            # Build status message with styling
            status_msg = f"""✅ DEFENSE ACTIVATED

🎯 Destruction Level: {destruction_level}%
{'🔥 NUCLEAR MODE ENGAGED' if destruction_level >= 80 else '⚡ Standard Protection' if destruction_level >= 50 else '👻 Stealth Mode'}
{'🎨 Targeted Areas Boosted 4x' if user_mask is not None else ''}
{'👤 Face Priority: ON' if face_boost else ''}

📊 Metrics:
{json.dumps(metrics, indent=2)}"""
            return output_path, heatmap_path, metrics, output_path, status_msg
        else:
            return None, None, None, None, "❌ Defense execution failed. Check logs."
            
    except Exception as ex:
        import traceback
        traceback.print_exc()
        return None, None, None, None, f"❌ Error: {str(ex)}"


# ==================== BUILD ENHANCED UI ====================

theme = gr.themes.Base(
    primary_hue="purple",
    secondary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter")
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_width="1px",
    block_border_color="*neutral_700",
    input_background_fill="*neutral_800",
    button_primary_background_fill="*primary_600",
)

with gr.Blocks(title="Project Invisible | AI Defense Command Center") as demo:
    
    # ==================== HEADER ====================
    gr.Markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <h1 style="font-size: 2.5em; margin-bottom: 0;">🛡️ PROJECT INVISIBLE</h1>
            <h3 style="font-weight: 300; opacity: 0.8; margin-top: 5px;">AI Identity Defense Command Center</h3>
            <p style="opacity: 0.6; font-size: 0.9em;">Latent Space Surgery • Adversarial Patches • Compliance Steganography</p>
        </div>
        """
    )
    
    with gr.Tabs() as tabs:
        
        # ===================== TAB 1: COMMAND CENTER =====================
        with gr.TabItem("⚔️ Defense Command Center", id="defense"):
            
            with gr.Row():
                # LEFT PANEL - Controls
                with gr.Column(scale=1):
                    gr.Markdown("### 🎛️ Control Panel")
                    
                    # Destruction Level
                    with gr.Group():
                        gr.Markdown("#### 💥 Destruction Level")
                        destruction_slider = gr.Slider(
                            minimum=0, maximum=100, value=75, step=1,
                            label="",
                            info="0 = Invisible Stealth | 50 = Balanced | 100 = Anti-Grok Nuclear"
                        )
                        destruction_display = gr.Markdown("**Current: 75% - Standard Protection**")
                        
                        def update_destruction_display(val):
                            if val >= 80:
                                return f"**Current: {val}% - 🔥 NUCLEAR MODE**"
                            elif val >= 50:
                                return f"**Current: {val}% - ⚡ Standard Protection**"
                            else:
                                return f"**Current: {val}% - 👻 Stealth Mode**"
                        
                        destruction_slider.change(
                            fn=update_destruction_display,
                            inputs=[destruction_slider],
                            outputs=[destruction_display]
                        )
                    
                    # Targeting Options
                    with gr.Group():
                        gr.Markdown("#### 🎯 Targeting Options")
                        
                        face_boost_toggle = gr.Checkbox(
                            label="👤 High-Priority Face Defense",
                            value=True,
                            info="4x gradient boost on facial features"
                        )
                        
                        targeting_mode = gr.Checkbox(
                            label="🎨 Draw-to-Destroy Mode",
                            value=False,
                            info="Draw on image to mark areas for nuclear protection"
                        )
                    
                    # Compliance
                    with gr.Group():
                        gr.Markdown("#### 📜 Compliance Shield")
                        compliance_layer = gr.Checkbox(
                            value=True,
                            label="📝 Inject GDPR Block Watermark",
                            info="Invisible steganography payload"
                        )
                        use_badge = gr.Checkbox(
                            value=True,
                            label="🛡️ Add Visual Trust Badge",
                            info="Overlays a clean 'Protected' shield icon"
                        )
                        badge_text = gr.Textbox(
                            value="PROTECTED",
                            label="📝 Badge Text",
                            placeholder="Enter text to display on badge...",
                            info="Custom text shown next to shield icon",
                            max_lines=1
                        )
                    
                    # Action Button
                    submit_btn = gr.Button(
                        "⚡ ACTIVATE DEFENSE",
                        variant="primary",
                        size="lg"
                    )
                
                # CENTER PANEL - Image Editor
                with gr.Column(scale=2):
                    gr.Markdown("### 📸 Target Image")
                    gr.Markdown("*Upload your image. Enable 'Draw-to-Destroy' to mark specific areas.*")
                    
                    input_editor = gr.ImageEditor(
                        label="",
                        type="numpy",
                        height=500,
                        brush=gr.Brush(
                            colors=["#FF0000", "#FF6600", "#FFFF00"],
                            default_size=30,
                            color_mode="fixed"
                        ),
                        eraser=gr.Eraser(default_size=30),
                        layers=True,
                        sources=["upload", "clipboard"],
                        interactive=True
                    )
                
                # RIGHT PANEL - Results
                with gr.Column(scale=2):
                    gr.Markdown("### 🔒 Protected Output")
                    
                    output_image = gr.Image(
                        label="Protected Image",
                        height=300
                    )
                    
                    with gr.Row():
                        heatmap_image = gr.Image(
                            label="🔥 Injection Heatmap",
                            height=200
                        )
                    
                    with gr.Accordion("📊 Defense Metrics", open=True):
                        metrics_json = gr.JSON(label="")
                    
                    download_btn = gr.File(label="📥 Download Protected Image")
                    
                    status_text = gr.Textbox(
                        label="📟 System Status",
                        lines=8,
                        interactive=False,
                        elem_classes=["status-box"]
                    )
            
            # Wire up the defense button
            submit_btn.click(
                fn=process_image_advanced,
                inputs=[input_editor, destruction_slider, face_boost_toggle, targeting_mode, compliance_layer, use_badge, badge_text],
                outputs=[output_image, heatmap_image, metrics_json, download_btn, status_text]
            )



    # ==================== FOOTER ====================
    gr.Markdown(
        """
        <div style="text-align: center; padding: 20px; opacity: 0.5; font-size: 0.9em;">
            <hr style="border-color: rgba(255,255,255,0.1);">
            <strong>Defense Stack:</strong> Stable Diffusion 1.5 • InsightFace ArcFace • CLIP ViT-B/32 • LPIPS
            <br>
            <em>Project Invisible © 2026 | Protecting Digital Identity</em>
        </div>
        """
    )

# ==================== LAUNCH ====================
if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        prevent_thread_lock=True,
        theme=theme,
        css=custom_css
    )
    
    logger.info("🛡️ Project Invisible Command Center running on port 8080...")
    while True:
        time.sleep(1)
