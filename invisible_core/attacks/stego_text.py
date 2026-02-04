
import numpy as np
from invisible_core.logger import logger
from PIL import Image
import cv2

# Import invisible-watermark
try:
    from imwatermark import WatermarkEncoder
except ImportError:
    logger.warning("Warning: 'invisible-watermark' library not found. StegoInjector will fail.")

class StegoInjector:
    """
    Project Invisible: Compliance Steganaography.
    Embeds invisible text payload into the frequency domain (DWT/DCT).
    """
    def __init__(self):
        self.encoder = None
        try:
            self.encoder = WatermarkEncoder()
            # 'dwtDct' is robust to compression and resizing
            self.method = 'dwtDct' 
            # self.encoder.set_watermark('bytes', self.method) # Removed redundant call causing init error
        except Exception as e:
            logger.error(f"[StegoInjector] Init failed: {e}")

    def inject_text(self, image: Image.Image, payload: str = "GDPR_BLOCK") -> Image.Image:
        """
        Embeds the payload string invisibly.
        """
        if not self.encoder:
            logger.warning("[StegoInjector] Encoder not ready. Returning original.")
            return image
            
        logger.info(f"[StegoInjector] Injecting payload '{payload}' using {self.method}...")
        
        # Convert to BGR for library
        img_np = np.array(image)
        if img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Payload to bytes
        # imwatermark expects bytes usually
        payload_bytes = payload.encode('utf-8')
        
        try:
            self.encoder.set_watermark('bytes', payload_bytes)
            encoded_bgr = self.encoder.encode(img_bgr, self.method)
            
            # Encoded is BGR
            encoded_rgb = cv2.cvtColor(encoded_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(encoded_rgb)
            
        except Exception as e:
            logger.error(f"[StegoInjector] Injection failed: {e}")
            return image
