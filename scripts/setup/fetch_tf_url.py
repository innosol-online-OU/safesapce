
import requests
import sys

def get_tf_url():
    print("Fetching TensorFlow metadata...")
    try:
        r = requests.get("https://pypi.org/pypi/tensorflow/json", timeout=10)
        if r.status_code != 200:
            print(f"Error fetching metadata: {r.status_code}")
            return

        data = r.json()
        releases = data.get("releases", {})
        
        # We need the version that matches what pip was trying to dl.
        # It said 2.20.0? Let's check if 2.20.0 exists.
        # If not, find the latest stable.
        
        target_version = None
        if "2.16.1" in releases: target_version = "2.16.1" # Common stable
        if "2.15.0" in releases: target_version = "2.15.0"
        
        # Check specific versions mentioned in logs if possible, but 2.20 seems wrong for TF? 
        # (Current TF is 2.16-2.17 range).
        # TensorBOARD is at 2.16/2.17 too.
        
        # Let's just find the latest 'manylinux' wheel for cp311.
        for ver in sorted(releases.keys(), reverse=True):
             # Skip RCs
             if "rc" in ver or "b" in ver: continue
             
             files = releases[ver]
             for f in files:
                 if "cp311-cp311-" in f["filename"] and "manylinux" in f["filename"] and "x86_64" in f["filename"]:
                     print(f"Found candidate: {f['filename']}")
                     print(f"URL: {f['url']}")
                     return

    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    get_tf_url()
