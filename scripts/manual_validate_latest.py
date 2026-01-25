
import os
import glob
from pathlib import Path
import sys

# Add src to path
sys.path.append(os.getcwd())


from invisible_core.critics.qwen_critic import QwenCritic

def validate_latest():
    # Force use of local LM Studio
    os.environ["LM_STUDIO_URL"] = "http://127.0.0.1:1234/v1/chat/completions"
    
    upload_dir = "uploads" # Relative execution
    if not os.path.exists(upload_dir):
        # try abs path
        upload_dir = "d:/job_assignments/Innosol/safespace/uploads"
    
    # Get all .png files
    files = glob.glob(os.path.join(upload_dir, "*.png"))
    
    # Sort by modification time
    files.sort(key=os.path.getmtime, reverse=True)
    
    # Find latest cloaked
    cloaked_path = None
    input_path = None
    
    for f in files:
        if "cloaked_" in os.path.basename(f) and not "heatmap_" in os.path.basename(f):
            cloaked_path = f
            break
            
    if not cloaked_path:
        print("No cloaked image found.")
        return

    print(f"Latest Output: {os.path.basename(cloaked_path)}")
    
    # Find corresponding input?
    # cloaked_UUID.png.
    # input_UUID.png might NOT match if uuid was generated separately.
    # app.py: input_filename = f"{uuid.uuid4()}_input.png"
    # output_filename = f"cloaked_{uuid.uuid4().hex}.png"
    # They have DIFFERENT UUIDs.
    
    # So I must find the latest input file that is OLDER than the cloaked file?
    # Or just the latest input file overall?
    
    # Let's find latest input file.
    for f in files:
        if "_input.png" in os.path.basename(f):
            input_path = f
            break
            
    print(f"Latest Input: {os.path.basename(input_path)}")
    
    print("\nRunning Qwen Validation...")
    critic = QwenCritic()
    
    # Pairwise or Single? Live mode uses PW if ref available.
    passed, reason, score = critic.critique_pairwise(input_path, cloaked_path)
    
    print(f"\nRESULT: {passed}")
    print(f"REASON: {reason}")
    print(f"SCORE: {score}")

if __name__ == "__main__":
    validate_latest()
