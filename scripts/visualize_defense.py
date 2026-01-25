"""
Project Invisible - Visual Defense Report Generator
Generates diagnostic image proving Phase 4.7 attack successfully blinded the AI.
"""

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO


def generate_report(original_path: str, protected_path: str, output_path: str = "defense_report.png"):
    """
    Generate a visual report comparing original and protected images.
    
    Args:
        original_path: Path to original image
        protected_path: Path to protected (Phase 4.7) image
        output_path: Path to save the report
    """
    print("[Report] Loading images...")
    
    # 1. Load Images
    img_orig = cv2.imread(original_path)
    img_prot = cv2.imread(protected_path)
    
    if img_orig is None:
        raise FileNotFoundError(f"Original image not found: {original_path}")
    if img_prot is None:
        raise FileNotFoundError(f"Protected image not found: {protected_path}")
    
    # Ensure same size
    img_prot = cv2.resize(img_prot, (img_orig.shape[1], img_orig.shape[0]))
    
    # 2. Compute "Warp Map" (The Geometric Difference)
    # Calculate absolute difference and amplify it 50x so humans can see it
    diff = cv2.absdiff(img_orig, img_prot)
    diff_amplified = np.clip(diff * 50, 0, 255).astype(np.uint8)
    
    # 3. Get AI Vision (YOLO Segmentation)
    print("[Report] Running YOLO inference...")
    model = YOLO("yolov8n-seg.pt")
    
    # Run Inference on BOTH
    res_orig = model(img_orig, verbose=False)[0]
    res_prot = model(img_prot, verbose=False)[0]
    
    # 4. Extract "Person" Masks (Class 0)
    def get_mask_overlay(res, img_shape):
        if res.masks is None or len(res.masks) == 0:
            return np.zeros((img_shape[0], img_shape[1]), dtype=np.float32)
        
        # Combine all "Person" masks
        combined_mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.float32)
        for idx, cls in enumerate(res.boxes.cls.cpu().numpy()):
            if int(cls) == 0:  # Class 0 = Person
                # Resize mask to image size
                m = res.masks.data[idx].cpu().numpy()
                m = cv2.resize(m, (img_shape[1], img_shape[0]))
                combined_mask = np.maximum(combined_mask, m)
        return combined_mask
    
    mask_orig = get_mask_overlay(res_orig, img_orig.shape)
    mask_prot = get_mask_overlay(res_prot, img_orig.shape)
    
    # Get confidence scores
    orig_conf = 0.0
    prot_conf = 0.0
    
    for idx, cls in enumerate(res_orig.boxes.cls.cpu().numpy()):
        if int(cls) == 0:
            orig_conf = max(orig_conf, res_orig.boxes.conf[idx].cpu().item())
    
    for idx, cls in enumerate(res_prot.boxes.cls.cpu().numpy()):
        if int(cls) == 0:
            prot_conf = max(prot_conf, res_prot.boxes.conf[idx].cpu().item())
    
    # 5. Plot the Dashboard
    print("[Report] Generating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Project Invisible - Phase 4.7 Defense Report", fontsize=16, fontweight='bold')
    
    # Row 1: Visuals
    axes[0,0].imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title("Original Image (Human View)")
    axes[0,0].axis('off')
    
    axes[0,1].imshow(cv2.cvtColor(img_prot, cv2.COLOR_BGR2RGB))
    axes[0,1].set_title("Protected Image (Phase 4.7)")
    axes[0,1].axis('off')
    
    axes[0,2].imshow(diff_amplified)
    axes[0,2].set_title("The 'Warp Map' (Difference x50)")
    axes[0,2].axis('off')
    
    # Row 2: AI Vision (The Truth)
    im1 = axes[1,0].imshow(mask_orig, cmap='jet', vmin=0, vmax=1)
    axes[1,0].set_title(f"AI Vision: BEFORE (Conf={orig_conf:.2f})")
    axes[1,0].axis('off')
    
    im2 = axes[1,1].imshow(mask_prot, cmap='jet', vmin=0, vmax=1)
    axes[1,1].set_title(f"AI Vision: AFTER (Conf={prot_conf:.2f})")
    axes[1,1].axis('off')
    
    # Row 2, Col 3: Diagnostics Text
    status = "SUCCESS" if prot_conf < 0.1 else ("PARTIAL" if prot_conf < 0.5 else "FAILED")
    status_color = "green" if status == "SUCCESS" else ("orange" if status == "PARTIAL" else "red")
    
    axes[1,2].axis('off')
    diag_text = (
        f"DIAGNOSTICS:\n\n"
        f"Original Person Conf: {orig_conf:.4f}\n"
        f"Protected Person Conf: {prot_conf:.4f}\n\n"
        f"Conf Reduction: {(1 - prot_conf/max(orig_conf, 1e-5))*100:.1f}%\n\n"
        f"Status: {status}"
    )
    axes[1,2].text(0.1, 0.5, diag_text, fontsize=14, 
                   transform=axes[1,2].transAxes,
                   verticalalignment='center',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"[Report] Generated: {output_path}")
    print(f"[Report] Original Person Conf: {orig_conf:.4f}")
    print(f"[Report] Protected Person Conf: {prot_conf:.4f}")
    print(f"[Report] Status: {status}")
    
    return orig_conf, prot_conf, status


if __name__ == "__main__":
    import sys
    
    # Check command line args
    if len(sys.argv) >= 3:
        original = sys.argv[1]
        protected = sys.argv[2]
        output = sys.argv[3] if len(sys.argv) > 3 else "defense_report.png"
    else:
        # Auto-detect from uploads folder
        upload_dir = "uploads"
        if os.path.exists(upload_dir):
            files = sorted([f for f in os.listdir(upload_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            if len(files) >= 2:
                original = os.path.join(upload_dir, files[-2])
                protected = os.path.join(upload_dir, files[-1])
                output = "defense_report.png"
                print(f"[Report] Auto-detected files:")
                print(f"         Original: {original}")
                print(f"         Protected: {protected}")
            else:
                print("Usage: python visualize_defense.py <original.png> <protected.png> [output.png]")
                sys.exit(1)
        else:
            print("Usage: python visualize_defense.py <original.png> <protected.png> [output.png]")
            sys.exit(1)
    
    generate_report(original, protected, output)
