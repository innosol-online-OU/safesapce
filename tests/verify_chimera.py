
import sys
import os
import torch
import cv2
import numpy as np
from PIL import Image

sys.path.append("/app")

def test_chimera():
    print("Testing Chimera Mask / YOLO...")
    try:
        from ultralytics import YOLO
        print("Ultralytics imported successfully.")
        
        model_path = "models/yolov8n-seg.pt"
        if not os.path.exists(model_path):
            print(f"Downloading model to {model_path}...")
            # YOLO class usually auto-downloads, but let's see.
        
        model = YOLO(model_path)
        print(f"Model {model_path} loaded.")
        
        # Try to load a real image (dummy_input.png seems large enough to be real)
        if os.path.exists("dummy_input.png"):
            img_path = "dummy_input.png"
        else:
            print("dummy_input.png not found, using blank.")
            img_path = "test_chimera.png"
            Image.new('RGB', (512, 512), color='white').save(img_path)
            
        print(f"Running inference on {img_path}...")
        results = model(img_path, verbose=True)
        
        print(f"Detected {len(results)} results.")
        if len(results) > 0:
            print(f"Boxes: {len(results[0].boxes)}")
            if results[0].masks:
                print(f"Masks: {results[0].masks.shape}")
            else:
                print("No masks found (maybe no person?)")
                
        print("SUCCESS: Chimera Mask pipeline is functional.")
        
    except ImportError as e:
        print(f"FAIL: ImportError - {e}")
    except Exception as e:
        print(f"FAIL: Runtime Error - {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chimera()
