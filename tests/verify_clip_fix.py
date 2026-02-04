
import sys
sys.path.append("/app")
from invisible_core.critics.clip_critic import CLIPCritic

print("Testing CLIPCritic loading with SafeTensors...")
try:
    critic = CLIPCritic(device='cpu') # CPU is fine for load test
    critic.load()
    print("SUCCESS: CLIPCritic loaded without CVE error.")
except Exception as e:
    print(f"FAIL: {e}")
    exit(1)
