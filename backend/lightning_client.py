"""
Lightning AI Client for Cloud Inference.
Handles job submission and status polling for remote GPU execution.
"""
import os
import shutil
import time
import threading
import uuid
import logging
from typing import Optional, Dict, Any
import zipfile

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    from lightning_sdk import Studio, Job
except ImportError as e:
    logger.warning(f"Lightning SDK not available: {e}")
    Studio = None
    Job = None

class LightningClient:
    def __init__(self, user_id: str = None, api_key: str = None):
        self.user_id = user_id or os.getenv("LIGHTNING_USER_ID", "")
        self.api_key = api_key or os.getenv("LIGHTNING_API_KEY")
        self.studio_name = os.getenv("LIGHTNING_STUDIO_NAME")
        self.team = os.getenv("LIGHTNING_TEAM")
        self.project = os.getenv("LIGHTNING_PROJECT")
        self.machine = os.getenv("LIGHTNING_MACHINE", "T4_SMALL")
        self.hf_token = os.getenv("HF_TOKEN")

        # Dynamic Power Management
        self.last_activity = time.time()
        self.idle_timeout = int(os.getenv("LIGHTNING_IDLE_TIMEOUT", 1800))  # 30 mins default
        self.monitor_thread = None

        # Initialize SDK if available
        if self.api_key and Studio:
            logger.info(f"Initialized for {self.user_id} @ {self.studio_name}")

    def start_monitor(self):
        """Start background thread to monitor idleness."""
        if self.monitor_thread is None:
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Power management monitor started")

    def _monitor_loop(self):
        """Background loop to stop Studio when idle."""
        while True:
            time.sleep(60)  # Check every minute
            if not self.is_configured() or not Studio:
                continue

            idle_time = time.time() - self.last_activity
            if idle_time > self.idle_timeout:
                try:
                    studio = Studio(self.studio_name, user=self.team, teamspace=self.project)
                    logger.info(f"Studio idle for {idle_time:.0f}s. Stopping to save credits...")
                    studio.stop()
                    logger.info("Studio stopped successfully")
                    self.last_activity = time.time()
                except Exception as e:
                    logger.error(f"Error stopping studio: {e}")

    def is_configured(self) -> bool:
        """Check if critical credentials are present."""
        valid = bool(self.api_key and self.studio_name and self.team and self.project)
        if not valid:
            logger.warning(
                f"Config incomplete - API_KEY={bool(self.api_key)}, "
                f"Studio={self.studio_name}, Team={self.team}, Project={self.project}"
            )
        return valid

    def submit_job(self, input_path: str, output_path: str, config: Dict[str, Any] = None, callback=None) -> Optional[str]:
        """
        Submit a protection job to the cloud.
        Supports callback(progress, message, url, credit_estimate) for real-time updates.
        """
        if not self.is_configured():
            return None

        logger.info("Submitting job to cloud...")
        if callback:
            callback(0, "Submitting to Cloud", None, 0.0)

        if not Studio:
            logger.error("Lightning SDK not installed")
            return None

        # Generate unique job context
        job_id = uuid.uuid4().hex[:8]
        runner_name = f"runner_{job_id}.py"
        payload_dir = f"payload_{job_id}"
        payload_zip = f"payload_{job_id}.zip"

        try:
            # Connect
            logger.info(f"Connecting to Studio: {self.studio_name}...")
            s = Studio(self.studio_name, user=self.team, teamspace=self.project)
            s.start(machine=self.machine)
            if callback:
                callback(5, "Studio connected", None, 0.0)

            # Upload Input
            s.run("mkdir -p uploads outputs")
            remote_input = f"uploads/{os.path.basename(input_path)}"
            logger.info(f"Uploading input to {remote_input}...")
            s.upload_file(input_path, remote_input)
            if callback:
                callback(10, "Input uploaded", None, 0.0)

            # Sync Code (Payload Isolation Strategy)
            logger.info("Syncing code with payload isolation...")

            # 1. Clean stale root-level folders (global cleanup)
            s.run("rm -rf invisible_core || true")
            s.run("pip uninstall -y invisible_core || true")

            # 2. Create local payload zip
            try:
                with zipfile.ZipFile(payload_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                    for root, dirs, files in os.walk("invisible_core"):
                        for file in files:
                            # Skip cache files
                            if "__pycache__" in root or file.endswith(".pyc"):
                                continue
                            file_path = os.path.join(root, file)
                            zf.write(file_path, file_path)

                # 3. Upload payload
                s.upload_file(payload_zip, payload_zip)

                # 4. Extract to unique directory
                s.run("which unzip || (sudo apt-get update && sudo apt-get install -y unzip)")
                s.run(f"rm -rf {payload_dir} || true")  # Clean this job's payload dir
                s.run(f"mkdir -p {payload_dir}")
                s.run(f"unzip -o {payload_zip} -d {payload_dir}")

                # 5. Verify payload structure (debug level)
                logger.debug("Verifying payload structure...")
                struct = s.run(f"ls -R {payload_dir}")
                logger.debug(struct)

            except Exception as e:
                logger.error(f"Code sync error: {e}")
                return None
            finally:
                if os.path.exists(payload_zip):
                    os.remove(payload_zip)

            if callback:
                callback(15, "Code synced", None, 0.0)

            # Config
            remote_output = f"outputs/{os.path.basename(output_path)}"
            config = config or {}
            strength_val = config.get('strength', 50)
            gm_grid = config.get('ghost_mesh_grid', 24)
            gm_balance = config.get('ghost_mesh_balance', 0.5)
            gm_anchoring = config.get('ghost_mesh_anchoring', 0.8)
            gm_tv = config.get('ghost_mesh_tv', 50)
            gm_jnd = config.get('ghost_mesh_jnd', True)

            # Generate Runner Script with Signature Handshake
            script_content = f"""
import sys
import os
import traceback
import inspect
import shutil

# --- PAYLOAD ISOLATION SETUP ---
PAYLOAD_DIR = "{payload_dir}"
payload_path = os.path.join(os.getcwd(), PAYLOAD_DIR)

# Force payload to take precedence
sys.path.insert(0, os.path.abspath(payload_path))
print(f'[Setup] CWD: {{os.getcwd()}}', flush=True)
print(f'[Setup] Payload path: {{payload_path}}', flush=True)

# --- PRE-FLIGHT SIGNATURE HANDSHAKE ---
try:
    import invisible_core
    print(f'[Pre-Flight] invisible_core loaded from: {{invisible_core.__file__}}', flush=True)
    
    from invisible_core.cloaking import CloakEngine
    sig = inspect.signature(CloakEngine.apply_defense)
    print(f'[Pre-Flight] apply_defense signature: {{sig}}', flush=True)
    
    # Verify Ghost-Mesh parameters exist
    required_params = ['ghost_mesh_grid', 'ghost_mesh_balance', 'ghost_mesh_anchoring', 
                       'ghost_mesh_tv', 'ghost_mesh_jnd']
    for param in required_params:
        if param not in sig.parameters:
            print(f'[ABORT] Missing required parameter: {{param}}', flush=True)
            sys.exit(1)
    
    print('[Pre-Flight] ✓ Signature handshake passed', flush=True)

except ImportError as e:
    print(f'[ABORT] Import error during pre-flight: {{e}}', flush=True)
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f'[ABORT] Pre-flight check failed: {{e}}', flush=True)
    traceback.print_exc()
    sys.exit(1)

# --- MAIN EXECUTION ---
try:
    engine = CloakEngine()
    
    success, heatmap, metrics = engine.apply_defense(
        input_path='{remote_input}',
        output_path='{remote_output}',
        strength={strength_val},
        target_profile='ghost_mesh',
        optimization_steps=50,
        is_liquid_v2={strength_val > 50},
        ghost_mesh_grid={gm_grid},
        ghost_mesh_balance={gm_balance},
        ghost_mesh_anchoring={gm_anchoring},
        ghost_mesh_tv={gm_tv},
        ghost_mesh_jnd={gm_jnd}
    )
    print(f'[Result] Done: {{success}}', flush=True)
    
except Exception as e:
    print(f'[ERROR] Execution failed: {{e}}', flush=True)
    traceback.print_exc()
    sys.exit(1)
"""

            # Write and Upload Runner Script
            local_script = f"runner_{job_id}.py"
            with open(local_script, "w") as f:
                f.write(script_content)

            s.upload_file(local_script, runner_name)
            if callback:
                callback(20, "Running...", None, 0.0)

            # Poller Setup
            stop_polling = False
            start_time = time.time()
            COST_PER_SEC = 0.15 / 3600.0

            def poller():
                last_step = 0
                while not stop_polling:
                    try:
                        elapsed = time.time() - start_time
                        cost = elapsed * COST_PER_SEC
                        if callback:
                            callback(None, None, None, cost)

                        # Check for intermediate steps
                        for step in range(last_step + 10, 60, 10):
                            remote_file = f"outputs/step_{step}.png"
                            local_file = f"uploads/step_{step}_{os.path.basename(input_path)}"
                            try:
                                s.download_file(remote_file, local_file)
                                if os.path.exists(local_file):
                                    if callback:
                                        callback(20 + step, f"Processing Step {step}...",
                                               f"/uploads/{os.path.basename(local_file)}", cost)
                                    last_step = step
                            except Exception:
                                pass  # File not ready yet
                        time.sleep(2)
                    except Exception as e:
                        logger.debug(f"Poller error: {e}")
                        time.sleep(5)

            t = threading.Thread(target=poller, daemon=True)
            t.start()

            # Execute Script
            logger.info("Running protection script...")
            if callback:
                callback(20, "Optimization Started...", None, None)

            # Set ENV vars for HF/Torch cache
            env_vars = (
                "export HF_TOKEN='None' && "
                "export HF_HOME='/teamspace/studios/this_studio/.cache/huggingface' && "
                "export TORCH_HOME='/teamspace/studios/this_studio/.cache/torch' && "
            )

            # Run
            try:
                result = s.run(f"{env_vars} python3 {runner_name}")
                success = "Done: True" in str(result)
            except Exception as e:
                logger.error(f"Execution error: {e}")
                result = str(e)
                success = False
            finally:
                stop_polling = True
                t.join()

            # Cleanup local temp file
            if os.path.exists(local_script):
                os.remove(local_script)

            # Cleanup remote artifacts
            try:
                logger.info("Cleaning up remote artifacts...")
                s.run(f"rm -f {runner_name}")
                s.run(f"rm -rf {payload_dir}")
                s.run(f"rm -f {payload_zip}")
            except Exception as e:
                logger.warning(f"Remote cleanup warning: {e}")

            # Result Handling
            if success:
                logger.info("Job completed successfully!")
                try:
                    s.download_file(remote_output, output_path)
                    return output_path
                except Exception as e:
                    logger.error(f"Download error: {e}")
                    return None
            else:
                logger.error(f"Job failed. Output: {result}")
                return None

        except Exception as e:
            logger.error(f"Job submission error: {e}")
            return None
