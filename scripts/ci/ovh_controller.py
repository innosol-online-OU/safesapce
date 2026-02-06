import ovh
import argparse
import time
import sys
import os

def get_client():
    # Credentials should be passed via environment variables for security
    # OVH_ENDPOINT, OVH_APPLICATION_KEY, OVH_APPLICATION_SECRET, OVH_CONSUMER_KEY
    try:
        client = ovh.Client(
            endpoint='ovh-eu', # Assuming EU based on IP/Region, can be parameterized
            application_key=os.environ['OVH_APP_KEY'],
            application_secret=os.environ['OVH_APP_SECRET'],
            consumer_key=os.environ['OVH_CONSUMER_KEY']
        )
        return client
    except Exception as e:
        print(f"❌ Failed to initialize OVH client: {e}")
        sys.exit(1)

def get_instance_status(client, project_id, instance_id):
    try:
        # Check instance details
        # GET /cloud/project/{serviceName}/instance/{instanceId}
        result = client.get(f'/cloud/project/{project_id}/instance/{instance_id}')
        return result['status'] # ACTIVE, SHUTOFF, BUILD, etc.
    except ovh.APIError as e:
        print(f"❌ API Error checking status: {e}")
        sys.exit(1)

def wait_for_status(client, project_id, instance_id, target_status, timeout=300):
    print(f"⏳ Waiting for instance to become {target_status} (Timeout: {timeout}s)...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        current_status = get_instance_status(client, project_id, instance_id)
        print(f"   Current status: {current_status}")
        if current_status == target_status:
            print(f"✅ Instance reached {target_status}!")
            return True
        time.sleep(10)
    print(f"❌ Timeout waiting for status {target_status}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='OVH Instance Controller')
    parser.add_argument('action', choices=['start', 'stop', 'status'], help='Action to perform')
    parser.add_argument('--project-id', required=True, help='OVH Cloud Project ID')
    parser.add_argument('--instance-id', required=True, help='OVH Instance ID')
    
    args = parser.parse_args()
    client = get_client()

    if args.action == 'status':
        status = get_instance_status(client, args.project_id, args.instance_id)
        print(f"Instance Status: {status}")

    elif args.action == 'start':
        current = get_instance_status(client, args.project_id, args.instance_id)
        if current == 'ACTIVE':
            print("ℹ️ Instance is already ACTIVE.")
        else:
            print(f"🚀 Starting instance {args.instance_id}...")
            try:
                client.post(f'/cloud/project/{args.project_id}/instance/{args.instance_id}/start')
            except ovh.APIError as e:
                print(f"❌ Failed to start instance: {e}")
                sys.exit(1)
            
            # Wait for it to be active
            wait_for_status(client, args.project_id, args.instance_id, 'ACTIVE')
            
            # Additional wait for SSH/Docker to boot
            print("⏳ Waiting 30s for services (SSH/Docker) to initialize...")
            time.sleep(30) 

    elif args.action == 'stop':
        current = get_instance_status(client, args.project_id, args.instance_id)
        if current == 'SHUTOFF':
            print("ℹ️ Instance is already STOPPED (SHUTOFF).")
        else:
            print(f"🛑 Stopping instance {args.instance_id}...")
            try:
                client.post(f'/cloud/project/{args.project_id}/instance/{args.instance_id}/stop')
            except ovh.APIError as e:
                print(f"❌ Failed to stop instance: {e}")
                sys.exit(1)
            
            wait_for_status(client, args.project_id, args.instance_id, 'SHUTOFF')

if __name__ == "__main__":
    main()
