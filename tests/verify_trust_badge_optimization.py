
import time
import numpy as np
from PIL import Image
import sys
import os

# Add root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from invisible_core.attacks.latent_cloak import LatentCloak

def run_benchmark():
    img_size = (2048, 2048)
    image = Image.fromarray(np.random.randint(0, 255, (img_size[1], img_size[0], 3), dtype=np.uint8))

    lc = LatentCloak(lite_mode=True)

    # 50 repetitions to fit in the region
    long_text = "This is a much longer text to test the performance of the embedding algorithm when dealing with larger payloads." * 50

    print(f"Testing with message length: {len(long_text)} chars")

    # Warmup
    lc.add_trust_badge(image, "warmup")

    start_time = time.time()
    iterations = 50
    for _ in range(iterations):
        lc.add_trust_badge(image, long_text)
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    print(f"Average time per call: {avg_time:.6f} seconds")

    if avg_time > 0.1:
         print("WARNING: Performance seems slow. Optimization might not be working or machine is slow.")
    else:
         print("SUCCESS: Performance is good.")

if __name__ == "__main__":
    try:
        run_benchmark()
    except ImportError:
        print("Skipping benchmark due to missing dependencies.")
