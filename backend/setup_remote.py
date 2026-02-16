import os
import sys
from lightning_sdk import Studio

print("Starting Setup Script (Hardcoded Verification)...")

# Exact values from successful manual test
STUDIO_NAME = "gpu-1"
TEAM_USER = "ahmedgamal-8mg0l"
PROJECT_SPACE = "deploy-model-project"

print(f"Connecting to Studio '{STUDIO_NAME}' (User: {TEAM_USER}, Teamspace: {PROJECT_SPACE})...")

try:
    # Use positional name, keyword user/teamspace
    s = Studio(STUDIO_NAME, user=TEAM_USER, teamspace=PROJECT_SPACE)
    print("Studio object created successfully.")
    
    print("Starting Studio (this may take time)...")
    s.start()
    print("Studio Started/Connected!")
    
    # Dependencies
    deps = "opencv-python-headless timm lpips scipy cryptography insightface onnxruntime-gpu bitsandbytes accelerate diffusers transformers ftfy regex"
    cmd = f"pip install {deps}"
    
    print(f"Running command: {cmd}")
    job = s.run(cmd, name="Setup-Dependencies")
    print(f"Job submitted: {job.name}")
    print("Setup Complete (Async).")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
