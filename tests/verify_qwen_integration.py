
import os
import sys
import requests
from PIL import Image

# Add src to path if needed, assuming /app/src is where code lives
sys.path.insert(0, "/app")

def verify_integration():
    print("--- Verifying Qwen VL Integration ---")
    
    # 1. Check Env Vars or default config
    lm_studio_url = os.getenv("LM_STUDIO_URL", "http://host.docker.internal:1234/v1/chat/completions")
    print(f"Target URL: {lm_studio_url}")

    # 2. Check Connectivity
    base_url = lm_studio_url.rsplit('/', 3)[0] # http://host:1234
    print(f"Checking base URL: {base_url}...")
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=5)
        print(f"Connectivity Check: {resp.status_code}")
        if resp.status_code == 200:
            print("✅ LM Studio is reachable!")
            models = resp.json()
            print(f"Available Models: {[m['id'] for m in models.get('data', [])]}")
        else:
            print(f"⚠️ LM Studio returned status {resp.status_code}")
    except Exception as e:
        print(f"❌ Failed to connect to LM Studio: {e}")
        print("Ensure Qwen VL is running and 'Start Server' is clicked in LM Studio.")
        print("Ensure 'Cross-Origin-Resource-Sharing (CORS)' is ON in LM Studio config.") 
        return

    # 3. Test QwenCritic directly
    print("\n--- Testing QwenCritic ---")
    try:
        from invisible_core.critics.qwen_critic import QwenCritic
        critic = QwenCritic()
        
        # Create dummy image
        img_path = "test_qwen.png"
        Image.new('RGB', (100, 100), color='red').save(img_path)
        
        print("Asking Qwen to describe a red square...")
        # Force API usage just in case
        critic.use_api = True 
        critic.lm_studio_url = lm_studio_url
        
        max_retries = 3
        for i in range(max_retries):
            success, reason, score = critic.critique(img_path, target_name="SomethingImpossible")
            print(f"Critique Result: Passed={success}, Reason='{reason}'")
            
            if "API Error" not in reason:
                print("✅ QwenCritic successfully received a response!")
                break
            else:
                 print(f"Retrying ({i+1}/{max_retries})...")

    except ImportError as e:
        print(f"❌ Import Error: {e}")
    except Exception as e:
        print(f"❌ Runtime Error: {e}")

if __name__ == "__main__":
    verify_integration()
