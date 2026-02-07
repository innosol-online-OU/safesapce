from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import uuid
import sys
import logging
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core logic (ensure dependencies are installed)
try:
    from invisible_core.cloaking import CloakEngine
except ImportError:
    # Fallback for dev without full deps
    CloakEngine = None

app = FastAPI(title="SafeSpace API", version="1.0.0")

# Security Config
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin") # Default for dev only
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")

# CORS
origins = [
    "http://localhost:5173",  # React Dev
    "http://localhost",       # Nginx
    "https://safespace.innosol.online" # Production
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for now, restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth (Simple Token for now)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str

def get_current_user(token: str = Depends(oauth2_scheme)):
    if token == SECRET_KEY:
        return User(username=ADMIN_USERNAME)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != ADMIN_USERNAME or form_data.password != ADMIN_PASSWORD:
         raise HTTPException(status_code=400, detail="Incorrect username or password")

    return {"access_token": SECRET_KEY, "token_type": "bearer"}


# Engine Singleton
engine_instance = None
def get_engine():
    global engine_instance
    if engine_instance is None and CloakEngine:
        print("Loading CloakEngine...")
        engine_instance = CloakEngine()
    return engine_instance

@app.get("/status")
def health_check():
    return {"status": "ok", "gpu_available": True} # Mock GPU check

@app.post("/protect")
async def protect_image(
    file: UploadFile = File(...),
    strength: int = Form(50), # 0-100 slider
    current_user: User = Depends(get_current_user)
):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # Save input
    input_filename = f"{uuid.uuid4()}_{file.filename}"
    input_path = os.path.join(upload_dir, input_filename)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_filename = f"protected_{uuid.uuid4()}.png"
    output_path = os.path.join(upload_dir, output_filename)

    # Map 'Strength' slider (0-100) to Engine Params
    # Low (0-30): Frontier Lite (Fast)
    # Med (30-70): Liquid Warp (Strong)
    # High (70-100): Ghost Mesh (Maximum)

    engine = get_engine()
    if not engine:
        return {"error": "Engine not initialized (missing deps?)"}

    try:
        target_profile = "liquid_17" # Default safe choice
        steps = 100

        if strength < 30:
            target_profile = "frontier"
            steps = 20
        elif strength > 80:
            target_profile = "ghost_mesh"
            steps = 60 # Heavy

        success, heatmap_path, metrics = engine.apply_defense(
            input_path=input_path,
            output_path=output_path,
            strength=strength, # Pass raw slider value
            target_profile=target_profile,
            optimization_steps=steps,
            visual_mode="latent_diffusion",
            compliance=True,
            is_liquid_v2=(strength > 50 and strength <= 80) # Use V2 for medium-high
        )

        if success:
             return {
                 "status": "success",
                 "output_url": f"/uploads/{output_filename}",
                 "metrics": metrics
             }
        else:
             raise HTTPException(status_code=500, detail="Protection failed")

    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.staticfiles import StaticFiles
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
