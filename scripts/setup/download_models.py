#!/usr/bin/env python3
"""
Model Weights Download Script
Run this after cloning to download required model weights.

Most models auto-download on first use, but this script ensures
everything is ready before running the app.
"""
import os
import sys

def main():
    print("=" * 50)
    print("Project Invisible - Model Weights Setup")
    print("=" * 50)
    
    os.makedirs("models", exist_ok=True)
    
    # 1. YOLO Segmentation Model (auto-downloads)
    print("\n[1/3] YOLO Segmentation Model...")
    try:
        from ultralytics import YOLO
        model_path = "models/yolov8n-seg.pt"
        if os.path.exists(model_path):
            print(f"  ✓ Already exists: {model_path}")
        else:
            print("  ↓ Downloading yolov8n-seg.pt...")
            # YOLO auto-downloads to current dir, then we move it
            model = YOLO("yolov8n-seg.pt")
            if os.path.exists("yolov8n-seg.pt"):
                os.rename("yolov8n-seg.pt", model_path)
            print(f"  ✓ Downloaded: {model_path}")
    except ImportError:
        print("  ⚠ ultralytics not installed, skipping YOLO")
    
    # 2. InsightFace (buffalo_l) - auto-downloads on first use
    print("\n[2/3] InsightFace (buffalo_l)...")
    try:
        import insightface
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("  ✓ InsightFace models ready")
    except Exception as e:
        print(f"  ⚠ InsightFace setup skipped: {e}")
    
    # 3. Hugging Face models (SigLIP, CLIP) - auto-download on first use
    print("\n[3/3] Hugging Face Models...")
    print("  ℹ SigLIP and CLIP will auto-download on first app run")
    print("  ℹ (~1.5GB, requires internet on first launch)")
    
    print("\n" + "=" * 50)
    print("✓ Model setup complete!")
    print("  Run: streamlit run app_interface/app.py")
    print("=" * 50)

if __name__ == "__main__":
    main()
