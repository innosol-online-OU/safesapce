
try:
    import cv2
    print("cv2: OK")
except ImportError:
    print("cv2: MISSING")

try:
    import ultralytics
    print("ultralytics: OK")
except ImportError:
    print("ultralytics: MISSING")

try:
    import pytorch_wavelets
    from pytorch_wavelets import DWTForward, DWTInverse
    print("pytorch_wavelets: OK")
except Exception as e:
    print(f"pytorch_wavelets: MISSING - Error: {e}")
    import traceback
    traceback.print_exc()
