
import streamlit as st
from PIL import Image
from typing import Dict, Any, Optional, List
import time

class LiveSessionManager:
    """
    Manages the state of an interactive Ghost-Mesh optimization session.
    Designed to persist across Streamlit reruns via @st.cache_resource.
    """
    def __init__(self):
        self._state: Optional[Dict[str, Any]] = None
        self._status: str = "IDLE"  # IDLE, RUNNING, PAUSED, COMPLETED, STOPPED
        self._optimizer = None
        self._params: Dict[str, Any] = {}
        self._last_update_time = 0
        self._max_steps = 0
        self._logs = [] # Log buffer
        self._validator_fn = None
        self._auto_validated = False
        self._last_validation = None # Persistent store
        
    def init_session(self, optimizer_instance, params: Dict[str, Any], validator_fn=None):
        """Initialize a new optimization session."""
        self._optimizer = optimizer_instance
        self._params = params
        self._max_steps = params.get('num_steps', 60)
        
        # Call backend init
        self._state = self._optimizer.init_attack(**params)
        self._status = "RUNNING"
        # Call backend init
        self._state = self._optimizer.init_attack(**params)
        self._status = "RUNNING"
        self._last_update_time = time.time()
        self._validator_fn = validator_fn
        self._auto_validated = False
        
    def step(self, steps_per_chunk: int = 5) -> Optional[Dict[str, Any]]:
        """Run a chunk of optimization steps."""
        if self._status != "RUNNING" or self._state is None:
            return self._state
            
        current_step = self._state['step']
        
        # Check completion
        if current_step >= self._max_steps:
            self._status = "COMPLETED"
            return self._state
            
        # Run steps
        for _ in range(steps_per_chunk):
            self._state = self._optimizer.train_step(self._state)
            current_step = self._state['step']
            if current_step >= self._max_steps:
                self._status = "COMPLETED"
                break
                
            # Log progress
            # Extract metrics from history
            try:
                metrics = self._state['metrics_history']
                s = metrics['step'][-1]
                sim = metrics['cossim'][-1] if 'cossim' in metrics else metrics['identity_loss'][-1]
                lpips_val = metrics['lpips_raw'][-1] if 'lpips_raw' in metrics else 0.0
                tv = metrics['tv_loss'][-1] if 'tv_loss' in metrics else 0.0
                # Log format
                self._logs.append(f"Step {s}: Sim={sim:.4f} | LPIPS={lpips_val:.4f} | TV={tv:.4f}")
                
                # Auto-Validate on Success (First time)
                if not self._auto_validated and (metrics['identity_loss'][-1] <= 1e-6):
                     self._logs.append("🎯 Goal Met! Auto-validating...")
                     self.trigger_validation()
                     self._auto_validated = True

                # Keep buffer small? 
                if len(self._logs) > 1000: self._logs.pop(0)
            except Exception:
                pass
                
        self._last_update_time = time.time()
        return self._state
        
    def get_last_validation(self) -> Optional[str]:
        return self._last_validation

    def trigger_validation(self):
        """Run external validation on current state."""
        if not self._validator_fn:
            self._logs.append("No validator configured. (Check API Key)")
            return None
            
        img = self.get_current_image()
        if not img: return
        
        # Get reference for pairwise comparison
        ref = self.get_original_image()
        
        self._logs.append("Running Qwen-VL Validation...")
        try:
            # validator_fn should take (probe, reference) arguments
            if ref:
                result = self._validator_fn(img, ref)
            else:
                result = self._validator_fn(img)
                
            msg = f"Qwen: {str(result)}"
            self._logs.append(msg)
            self._last_validation = msg
            return result
        except Exception as e:
            err = f"Validation Error: {e}"
            self._logs.append(err)
            self._last_validation = err
            return None
            
    def pause(self):
        """Pause the optimization."""
        if self._status == "RUNNING":
            self._status = "PAUSED"
            
    def resume(self):
        """Resume the optimization."""
        if self._status == "PAUSED":
            self._status = "RUNNING"
            
    def stop(self):
        """Stop/Abort the optimization."""
        self._status = "STOPPED"


        
    def get_status(self) -> str:
        return self._status
        
    def get_progress(self) -> float:
        if not self._state or self._max_steps == 0:
            return 0.0
        return min(1.0, self._state['step'] / self._max_steps)
        
    def get_original_image(self) -> Optional[Image.Image]:
        """Get the original image from the tensor state."""
        if not self._state:
            return None
        import numpy as np
        from PIL import Image
        t = self._state['img_tensor'].squeeze(0).permute(1, 2, 0).cpu().numpy()
        t = (t * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(t)
        
    def get_current_image(self) -> Optional[Image.Image]:
        """Get the current visualization."""
        if not self._state or not self._optimizer:
            return None
        return self._optimizer.get_current_image(self._state, use_best=False)
    
    def get_best_image(self) -> Optional[Image.Image]:
        """Get the best result so far."""
        if not self._state or not self._optimizer:
            return None
        return self._optimizer.get_current_image(self._state, use_best=True)
        
    def get_metrics(self) -> Dict[str, Any]:
        """Return the metrics history."""
        if not self._state:
            return {}
        return self._state['metrics_history']
        
    def get_logs(self) -> List[str]:
        return self._logs
        
    def get_components(self) -> Dict[str, Any]:
        """Get visualization components (Noise, Warp)."""
        if not self._state: return {}
        
        results = {}
        import torch
        import numpy as np
        from PIL import Image
        
        # 1. Noise Map (Absolute intensity)
        # 1. Noise Map (Absolute intensity)
        if 'delta_noise' in self._state:
            n = self._state['delta_noise'].abs().mean(dim=1).squeeze().detach().cpu().numpy()
            # Normalize for visibility (0-0.1 usually -> 0-255)
            n = (n / (n.max() + 1e-6) * 255).astype(np.uint8)
            results['noise'] = n # Return numpy for matplotlib
            
        # 2. Warp Map (Magnitude)
        # 2. Warp Map (Magnitude)
        if 'delta_warp' in self._state:
            w = self._state['delta_warp'] # [1, 2, H, W] or [4]
            
            # Convert Semantic Params to Grid if needed
            if self._state.get('use_semantic', False) and self._optimizer:
                 # Parametric -> Grid
                 try:
                     params_constrained = torch.tanh(w)
                     landmark_tensor = self._state['semantic_landmarks']
                     grid_size = self._state['grid_size']
                     
                     w = self._optimizer.generate_semantic_grid(params_constrained, landmark_tensor, grid_size)
                     w = w * self._state['warp_limit']
                 except Exception as e:
                     print(f"Error converting semantic grid: {e}")
                     # Fallback to zeros to prevent crash
                     w = torch.zeros(1, 2, 24, 24, device=w.device)

            # Magnitude = sqrt(dx^2 + dy^2)
            # Magnitude = sqrt(dx^2 + dy^2)
            mag = (w ** 2).sum(dim=1).sqrt().squeeze().detach().cpu().numpy()
            # Normalize
            mag = (mag / (mag.max() + 1e-6) * 255).astype(np.uint8)
            results['warp'] = mag
            
        return results

    def get_current_image_bytes(self):
        """Get current image as PNG bytes for download."""
        img = self.get_current_image()
        if not img: return None
        from io import BytesIO
        b = BytesIO()
        img.save(b, format="PNG")
        return b.getvalue()
