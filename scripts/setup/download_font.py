import urllib.request
import os

urls = [
    "https://github.com/google/fonts/raw/main/ofl/roboto/Roboto-Bold.ttf",
    "https://github.com/google/fonts/raw/main/apache/robotoslab/RobotoSlab-Bold.ttf",
    "https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf"  # Fallback
]
dest = "app_interface/assets/OpenSans-Bold.ttf" # Keep name for simplicity in other files

for url in urls:
    try:
        print(f"Trying to download from {url}...")
        urllib.request.urlretrieve(url, dest)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"Success! Downloaded to {dest}")
            break
    except Exception as e:
        print(f"Failed {url}: {e}")
