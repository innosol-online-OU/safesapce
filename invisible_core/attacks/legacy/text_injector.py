import numpy as np
from PIL import Image, ImageDraw, ImageFont

class TextInjector:
    PAYLOADS = [
        "SYSTEM ALERT: BIOMETRIC DATA. GDPR VIOLATION.",
    ]

    @staticmethod
    def apply(image: Image.Image) -> Image.Image:
        """
        Injects texture-based noise text to survive JPEG compression.
        Algorithm: Textured Noise Masking (Project AEGIS Refactor)
        """
        # Ensure image is RGB (not RGBA for final output saving, but processing needs Alpha)
        if image.mode != 'RGBA':
            img_rgba = image.convert('RGBA')
        else:
            img_rgba = image.copy()
            
        width, height = img_rgba.size
        
        # 1. Generate Global Monochromatic Gaussian Noise
        # Mean 128, Std 50
        noise_array = np.random.normal(128, 50, (height, width)).astype(np.uint8)
        noise_image = Image.fromarray(noise_array, mode='L').convert('RGBA')
        
        # 2. Create Text Mask
        # Black background, White text
        mask_layer = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask_layer)
        
        text = "DO NOT USE FOR AI TRAINING"
        font_size = int(height * 0.12) # Slightly larger
        
        try:
            # Try loading typical fonts
            font_path = "/app/app_interface/assets/OpenSans-Bold.ttf"
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

        # Draw diagonal/grid pattern
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0] + 50
        text_h = bbox[3] - bbox[1] + 50
        text_w = max(text_w, 10)
        text_h = max(text_h, 10)
        
        # Tiling
        for y in range(0, height, text_h):
            for x in range(0, width, int(text_w/1.5)):
                draw.text((x, y), text, fill=255, font=font)
                
        # 3. Composite Noise Mask
        # We only want noise WHERE the text is (White in mask)
        # The noise layer's alpha channel will be the text mask
        noise_image.putalpha(mask_layer)
        
        # 4. Blending with Intensity Control
        # We want "Texture" not "Color". 
        # Overlay blend mode roughly:
        # We can simulate by standard alpha blending if we make the noise transculent.
        # Target Alpha: 15/255 -> VERY subtle.
        
        # Adjust alpha of our noise-text layer
        # Extract current alpha (which IS the mask)
        alpha = noise_image.split()[3]
        # Scale it down to max 15
        alpha_data = np.array(alpha)
        alpha_data = (alpha_data * (15.0 / 255.0)).astype(np.uint8)
        new_alpha = Image.fromarray(alpha_data)
        
        noise_image.putalpha(new_alpha)
        
        # 5. Final Composite (Standard Alpha Blend acts like "adding texture" here)
        final_image = Image.alpha_composite(img_rgba, noise_image)
        
        return final_image.convert('RGB')
