
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

class FaceMasker:
    """
    Generates a soft binary mask for the face region using MediaPipe Face Mesh.
    Output: 0.0 to 1.0 float32 numpy array (Single Channel).
    """
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def get_mask(self, image: Image.Image) -> np.ndarray:
        """
        Extracts face mask from PIL Image.
        Returns: (H, W) float32 array in range [0, 1].
        """
        # Convert PIL to CV2 (BGR)
        img_np = np.array(image)
        if img_np.shape[2] == 4: # RGBA
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            
        H, W = img_np.shape[:2]
        results = self.face_mesh.process(img_np)

        mask = np.zeros((H, W), dtype=np.uint8)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get hull of the face mesh
                dataset_landmarks = []
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * W), int(lm.y * H)
                    dataset_landmarks.append((x, y))
                
                # Convex Hull to get boundary
                points = np.array(dataset_landmarks, np.int32)
                hull = cv2.convexHull(points)
                
                # Draw filled polygon (White)
                cv2.fillConvexPoly(mask, hull, 255)
                
        # Apply Gaussian Blur for soft edges (Blend)
        mask_blurred = cv2.GaussianBlur(mask, (15, 15), 0)
        
        # Normalize to 0-1 float
        return mask_blurred.astype(np.float32) / 255.0
