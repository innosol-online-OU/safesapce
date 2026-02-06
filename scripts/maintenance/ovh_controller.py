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
        print("✅ Active")
        return

    if status == 'SHELVED_OFFLOADED':
        print("🚀 Unshelving...")
        client.post(f'/cloud/project/{SERVICE_NAME}/instance/{INSTANCE_ID}/unshelve')
    elif status == 'SHUTOFF':
        print("⚡ Starting...")
        client.post(f'/cloud/project/{SERVICE_NAME}/instance/{INSTANCE_ID}/start')
    
    # Wait Loop (60m Max)
    for i in range(360): 
        new_status = get_status()
        if new_status == 'ACTIVE':
            print("✅ Up. Waiting 60s...")
            time.sleep(60) 
            return
        print(f"⏳ {new_status}...")
        time.sleep(10)
    
    print("❌ Timeout (>60m)")
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
