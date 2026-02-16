from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
import shutil
import os
import uuid
import sys
import logging
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core logic (ensure dependencies are installed)
try:
    from invisible_core.cloaking import CloakEngine
except ImportError:
    # Fallback for dev without full deps
    logging.warning("Invisible Core not found. Running in mock mode.")
    CloakEngine = None

app = FastAPI(title="SafeSpace API", version="1.0.0")

# Security Config
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin") # Default for dev only
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")

# CORS
origins = [
    "http://localhost:5173",  # React Dev
    "http://localhost:3000",  # React Prod
    "http://localhost",       # Nginx
    "https://safespace.innosol.online", # Production
    "http://safespace.innosol.online"
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
    # Check GPU
    import torch
    gpu = torch.cuda.is_available()
    
    # Determine GPU Type
    gpu_type = None
    if gpu:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown GPU"
        gpu_type = gpu_name
    
    # Check Compute Mode & Studio Status
    mode = "local"
    studio_status = "n/a"
    machine_type = "CPU" if not gpu else gpu_type
    
    try:
        from .lightning_client import LightningClient
        client = LightningClient()
        if client.is_configured():
            mode = "cloud"
            machine_type = client.machine
            # Studio status would require API call - simplified for now
            studio_status = "connected"
    except Exception as e:
        logging.debug(f"Lightning client check failed: {e}")
        
    return {
        "status": "ok", 
        "gpu_available": gpu,
        "gpu_type": machine_type,
        "engine": "Phase 18 (Ghost Mesh)", 
        "compute_mode": mode,
        "studio_status": studio_status
    }

from datetime import datetime
import json

# Encrypted History Storage
from .crypto_utils import encrypt_data, decrypt_data

HISTORY_FILE = "uploads/history.enc"  # Encrypted file extension

def load_history():
    """Load and decrypt history from encrypted file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            encrypted = f.read()
        if not encrypted:
            return []
        decrypted = decrypt_data(encrypted)
        return json.loads(decrypted)
    except Exception as e:
        print(f"[History] Load error: {e}")
        return []

def save_history(history: list):
    """Encrypt and save history to file."""
    try:
        data = json.dumps(history)
        encrypted = encrypt_data(data)
        with open(HISTORY_FILE, "w") as f:
            f.write(encrypted)
    except Exception as e:
        print(f"[History] Save error: {e}")

def save_history_record(record):
    """Add a new record to history."""
    history = load_history()
    history.insert(0, record)
    if len(history) > 100:
        history = history[:100]
    save_history(history)

@app.get("/history")
def get_history(current_user: User = Depends(get_current_user)):
    """Get history (authenticated)."""
    return load_history()

@app.delete("/history/{record_id}")
def delete_history_item(record_id: str, current_user: User = Depends(get_current_user)):
    """Delete a specific history item by ID."""
    history = load_history()
    original_len = len(history)
    history = [h for h in history if h.get("id") != record_id]
    
    if len(history) == original_len:
        raise HTTPException(status_code=404, detail="Record not found")
    
    save_history(history)
    return {"status": "deleted", "id": record_id}

# ... existing code ...

# Ghost Mesh Configuration
class GhostMeshConfig(BaseModel):
    grid_resolution: int = 24  # 12-48, step 4
    warp_noise_balance: float = 0.5  # 0-1
    tzone_anchoring: float = 0.8  # 0-1
    grain_control: float = 0.3  # 0-1
    ghost_masking: bool = True

# Global Ghost Mesh Config
GHOST_MESH_CONFIG = GhostMeshConfig()

@app.get("/config/ghost-mesh")
def get_ghost_mesh_config():
    """Get current Ghost Mesh configuration."""
    return GHOST_MESH_CONFIG.model_dump()

@app.post("/config/ghost-mesh")
def update_ghost_mesh_config(config: GhostMeshConfig, current_user: User = Depends(get_current_user)):
    """Update Ghost Mesh configuration (authenticated)."""
    global GHOST_MESH_CONFIG
    GHOST_MESH_CONFIG = config
    logging.info(f"Ghost Mesh config updated: {config.model_dump()}")
    return {"status": "updated", "config": GHOST_MESH_CONFIG.model_dump()}

# Job Management
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ProtectionJob(BaseModel):
    id: str
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    message: str = "Queued"
    intermediate_url: Optional[str] = None
    output_url: Optional[str] = None
    heatmap_url: Optional[str] = None
    cost_estimate: float = 0.0
    metrics: Dict[str, Any] = {}
    created_at: str

JOBS: Dict[str, ProtectionJob] = {}

def update_job_callback(job_id: str, progress: int = None, message: str = None, 
                        intermediate_url: str = None, cost: float = None):
    """Callback for LightningClient to update job status."""
    if job_id in JOBS:
        if progress is not None: JOBS[job_id].progress = progress
        if message is not None: JOBS[job_id].message = message
        if intermediate_url is not None: JOBS[job_id].intermediate_url = intermediate_url
        if cost is not None: JOBS[job_id].cost_estimate = cost

async def process_protection_job(job_id: str, input_path: str, output_path: str, 
                                 strength: int, config: GhostMeshConfig, input_ref: str, output_ref: str):
    """Background task to run protection."""
    job = JOBS[job_id]
    job.status = JobStatus.RUNNING
    job.message = "Initializing..."
    
    try:
        from .lightning_client import LightningClient
        client = LightningClient()
        success = False
        
        # 1. Try Cloud Execution
        if client.is_configured():
            job.message = "Starting Cloud Execution..."
            
            def callback(prog, msg, url, credit):
                update_job_callback(job_id, prog, msg, url, credit)
            
            cloud_config = {
                "strength": strength,
                "ghost_mesh_grid": config.grid_resolution,
                "ghost_mesh_balance": config.warp_noise_balance,
                "ghost_mesh_anchoring": config.tzone_anchoring,
                "ghost_mesh_tv": config.grain_control * 100,
                "ghost_mesh_jnd": config.ghost_masking
            }
            
            # Pass callback to submit_job
            cloud_result = client.submit_job(input_path, output_path, cloud_config, callback=callback)
            
            if cloud_result and os.path.exists(cloud_result):
                success = True
                job.output_url = f"/uploads/{output_ref}"
                job.metrics = {"device": "cloud-gpu-T4", "strategy": "ghost-mesh-cloud"}
                
                # History
                record = {
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                    "original": f"/uploads/{input_ref}",
                    "protected": f"/uploads/{output_ref}",
                    "metrics": job.metrics,
                    "config": {"strength": strength, "phase": "Ghost-Mesh", "mode": "cloud"}
                }
                save_history_record(record)
            else:
                job.message = "Cloud failed. Falling back..."
        
        # 2. Local Fallback (DISABLED)
        if not success:
            raise Exception("Cloud Execution Failed. Local Fallback is DISABLED for debugging.")
            job.message = "Running Locally..."
            job.progress = 10
            
            engine = get_engine()
            if not engine: raise Exception("Engine not loaded")
            
            # ... phase 18 logic ...
            target_profile = "ghost_mesh"
            steps = 50
            if strength < 30: steps = 30 
            elif strength > 80: steps = 80 
            
            success_local, heatmap_path, metrics = engine.apply_defense(
                input_path=input_path,
                output_path=output_path,
                strength=strength,
                target_profile=target_profile,
                optimization_steps=steps,
                visual_mode="latent_diffusion",
                compliance=True,
                is_liquid_v2=(strength > 50),
                ghost_mesh_grid=config.grid_resolution,
                ghost_mesh_balance=config.warp_noise_balance,
                ghost_mesh_anchoring=config.tzone_anchoring,
                ghost_mesh_tv=config.grain_control * 100,
                ghost_mesh_jnd=config.ghost_masking
            )
            
            if success_local:
                success = True
                job.output_url = f"/uploads/{output_ref}"
                if heatmap_path and os.path.exists(heatmap_path):
                    job.heatmap_url = f"/uploads/{os.path.basename(heatmap_path)}"
                job.metrics = metrics
                
                record = {
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                    "original": f"/uploads/{input_ref}",
                    "protected": f"/uploads/{output_ref}",
                    "metrics": metrics,
                    "config": {"strength": strength, "phase": "Ghost-Mesh", "mode": "local"}
                }
                save_history_record(record)
        
        if success:
            job.status = JobStatus.COMPLETED
            job.progress = 100
            job.message = "Protection Complete"
        else:
            job.status = JobStatus.FAILED
            job.message = "Protection Failed"
            
    except Exception as e:
        job.status = JobStatus.FAILED
        job.message = f"Error: {str(e)}"
        logging.error(f"Job {job_id} failed: {e}")


@app.post("/protect")
async def protect_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    strength: int = Form(50), 
    current_user: User = Depends(get_current_user)
):
    # Setup Paths
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # Save Input
    input_ref = f"{uuid.uuid4()}_{file.filename}"
    input_path = os.path.join(upload_dir, input_ref)
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_ref = f"protected_{uuid.uuid4()}.png"
    output_path = os.path.join(upload_dir, output_ref)
    
    # Create Job
    job_id = str(uuid.uuid4())
    job = ProtectionJob(id=job_id, created_at=datetime.now().isoformat())
    JOBS[job_id] = job
    
    # Spawn Background Task
    background_tasks.add_task(
        process_protection_job, 
        job_id, input_path, output_path, strength, GHOST_MESH_CONFIG,
        input_ref, output_ref
    )

    return {"job_id": job_id, "status": "queued"}

@app.get("/jobs/{job_id}", response_model=ProtectionJob)
def get_job_status(job_id: str, current_user: User = Depends(get_current_user)):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]


from fastapi.staticfiles import StaticFiles
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
