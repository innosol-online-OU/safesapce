
import os
import urllib.request
from src.core.cloaking import CloakEngine
from validator import Validator

def download_face():
    path = "test_face.jpg"
    if not os.path.exists(path):
        print("Downloading test face (Barack Obama)...")
        # Use a reliable public domain face
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/President_Barack_Obama.jpg/480px-President_Barack_Obama.jpg"
        try:
            # Fake user agent to avoid 403
            req = urllib.request.Request(
                url, 
                data=None, 
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
                }
            )
            with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
                out_file.write(response.read())
        except Exception as e:
            print(f"Failed to download image: {e}")
            # Create dummy image if download fails for offline testing
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (512, 512), color = (73, 109, 137))
            d = ImageDraw.Draw(img)
            d.text((10,10), "Face Mock", fill=(255,255,0))
            img.save(path)
            
    return path

def main():
    img_path = download_face()
    out_path = "protected_invisible.png"
    
    print("\n--- [Phase 1] Running Project Invisible Engine ---")
    engine = CloakEngine()
    # First run might take time to load SD 2.1
    engine.apply_defense(img_path, out_path, visual_mode="latent_diffusion", compliance=True)
    
    if os.path.exists(out_path):
        print("\n--- [Phase 2] Running Validator (Red Team) ---")
        val = Validator()
        # "A photo of Barack Obama" is the prompt to check CLIP against
        success = val.validate(img_path, out_path, identity_text="A photo of Barack Obama")
        
        if success:
            print("\n>>> 🛡️ SYSTEM VERIFIED: PROJECT INVISIBLE IS SECURE <<<")
        else:
            print("\n>>> ⚠️ SYSTEM WARNING: DEFENSE DID NOT MEET INVISIBILITY STANDARDS <<<")
    else:
        print("❌ Error: Protected image not generated.")

if __name__ == "__main__":
    main()
