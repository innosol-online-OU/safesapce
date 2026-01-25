
import os
import sys
from PIL import Image
import numpy as np
from invisible_core.cloaking import CloakEngine

def check_similarity():
    engine = CloakEngine()
    
    # Create dummy input
    input_path = "debug_input.png"
    img = Image.new('RGB', (512, 512), color = (100, 100, 100))
    # Add some noise so it's not flat
    arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(input_path)
    
    out_privacy = "debug_privacy.png"
    out_art = "debug_art.png"
    
    print("Generating Privacy...")
    engine.apply_defense(input_path, out_privacy, visual_mode="privacy", compliance=False)
    
    print("Generating Art...")
    engine.apply_defense(input_path, out_art, visual_mode="art", compliance=False)
    
    # Compare
    img_p = Image.open(out_privacy)
    img_a = Image.open(out_art)
    
    diff = np.mean(np.abs(np.array(img_p) - np.array(img_a)))
    print(f"Mean Pixel Difference between Privacy and Art: {diff}")
    
    if diff == 0:
        print("CRITICAL: Images are IDENTICAL.")
    else:
        print("Images are distinct.")

if __name__ == "__main__":
    check_similarity()
