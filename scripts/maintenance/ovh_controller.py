import ovh
import sys
import time
import os

# --- CONFIGURATION ---
# We read credentials from Environment Variables for security
try:
    client = ovh.Client(
        endpoint='ovh-eu',               # Change to 'ovh-ca' or 'ovh-us' if outside EU
        application_key=os.environ['OVH_APP_KEY'],
        application_secret=os.environ['OVH_APP_SECRET'],
        consumer_key=os.environ['OVH_CONSUMER_KEY']
    )
    SERVICE_NAME = os.environ['OVH_PROJECT_ID']
    INSTANCE_ID = os.environ['OVH_INSTANCE_ID']
except KeyError as e:
    print(f"❌ Error: Missing Environment Variable {e}")
    sys.exit(1)

def get_status():
    try:
        instance = client.get(f'/cloud/project/{SERVICE_NAME}/instance/{INSTANCE_ID}')
        return instance['status']
    except ovh.exceptions.APIError as e:
        print(f"❌ OVH API Error: {e}")
        sys.exit(1)

def wake_up():
    status = get_status()
    print(f"🔍 Current Status: {status}")

    if status == 'ACTIVE':
        print("✅ Server is already running.")
        return

    # CASE 1: Deep Sleep (Shelved/Suspended) -> This saves money!
    if status == 'SHELVED_OFFLOADED':
        print("🚀 Unshelving (Waking from deep sleep)... This takes ~2-3 mins.")
        client.post(f'/cloud/project/{SERVICE_NAME}/instance/{INSTANCE_ID}/unshelve')
    
    # CASE 2: Soft Sleep (Stopped) -> This costs money!
    elif status == 'SHUTOFF':
        print("⚡ Starting (Waking from soft sleep)...")
        client.post(f'/cloud/project/{SERVICE_NAME}/instance/{INSTANCE_ID}/start')
    
    # Wait Loop
    # Unshelving can take time depending on disk size and OVH load.
    # We allow up to 15 minutes (90 * 10s).
    for i in range(90): 
        new_status = get_status()
        if new_status == 'ACTIVE':
            print("✅ Server is UP! Waiting 60s for network/docker...")
            time.sleep(60) 
            return
        print(f"⏳ Booting... ({new_status})")
        time.sleep(10)
    
    print("❌ Error: Server took too long to wake up (Timeout > 15m).")
    sys.exit(1)

def sleep_now():
    status = get_status()
    # If it is already shelved, we are good.
    if status == 'SHELVED_OFFLOADED':
        print("✅ Server is already Suspended (Free).")
        return

    print("🛑 Suspending (Shelving) server to stop billing...")
    # 'shelve' is the API command for 'Suspend'
    client.post(f'/cloud/project/{SERVICE_NAME}/instance/{INSTANCE_ID}/shelve')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ovh_controller.py [wake|sleep]")
        sys.exit(1)
        
    action = sys.argv[1]
    if action == "wake":
        wake_up()
    elif action == "sleep":
        sleep_now()
