# Contributing to SafeSpace

Thank you for your interest in contributing to SafeSpace! This guide covers how researchers and developers can contribute to the project.

---

## 🧪 For Researchers

### Research Workflow

SafeSpace is designed for **mixed research and product development**. Here's how to contribute:

#### 1. Setting Up Your Environment

```bash
# Clone and build
git clone https://github.com/YOUR_ORG/safespace.git
cd safespace
docker build -t safespace:latest .

# Run in development mode (mount local code)
docker run -it --gpus all \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/research:/app/research \
  -p 8501:8501 \
  safespace:latest bash
```

#### 2. Adding New Attack Methods

New attacks should be added to `src/core/protocols/`:

1. **Create your method** in `src/core/protocols/latent_cloak.py`:
   ```python
   def protect_your_method(self, image_path: str, **kwargs):
       """
       Phase X: Your Novel Attack
       """
       # Your implementation
       pass
   ```

2. **Update routing** in `src/core/cloaking.py`:
   ```python
   if target_profile == "your_method":
       return self.latent_cloak.protect_your_method(...)
   ```

3. **Add UI controls** in `src/app.py`:
   ```python
   target_profile_options = {
       "Your Method (Phase X)": "your_method",
       # ... existing methods
   }
   ```

#### 3. Research Experiments

The `research/` folder is structured for experimentation:

**Notebooks (`research/notebooks/`):**
- Exploratory data analysis
- Ablation studies
- Visualization of attack effects

**Training (`research/training/`):**
- Model training scripts
- Hyperparameter search
- Custom surrogate model training

**Experiments (`research/experiments/`):**
- One-off experiments
- Comparative studies
- Novel attack prototypes

Example notebook structure:
```python
# research/notebooks/phase_X_experiment.ipynb
1. Load SafeSpace components
2. Run attack with varying parameters
3. Evaluate metrics (SSIM, CLIP similarity, etc.)
4. Visualize results
5. Document findings
```

#### 4. Testing Protocol

Create a verification script in `tests/`:

```python
# tests/verify_your_method.py
from src.core.protocols.latent_cloak import LatentCloak

def verify_your_method():
    cloak = LatentCloak()
    result = cloak.protect_your_method("test.png", strength=50)
    
    # Assertions
    assert result is not None
    # ... more checks
    
if __name__ == "__main__":
    verify_your_method()
```

Run all verification:
```bash
python tests/verify_your_method.py
```

---

## 🌿 Branch Strategy

We use a three-branch model:

### Branches

- **`main`**: Production-ready code. All methods work without crashes.
- **`dev`**: Integration branch. Features merged from `research` after validation.
- **`research`**: Experimental features. May contain bugs or incomplete work.

### Workflow

1. **Create feature branch from `research`:**
   ```bash
   git checkout research
   git pull
   git checkout -b feature/phase-X-attack
   ```

2. **Develop and test:**
   - Add code to `src/core/protocols/`
   - Create verification script in `tests/`
   - Document in notebook if applicable

3. **Open PR to `research`:**
   - Include verification results
   - Document any new dependencies

4. **Validation (Private CI):**
   - Apply the `safe-to-test` label to your PR.
   - The **SafeSpace Bridge** will mirror your code to our Private CI (Gitea).
   - Check the **"Gitea/Fast-Feedback"** status on your PR (must be ✅).
   - If requested, check **"Gitea/Heavy-Validation"** (runs on GPU server).

5. **Merge:**
   - Once checks pass and code is reviewed, merge to `dev`.
   - `dev` changes are automatically synced to the internal factory.

---

## 📚 Documentation Standards

### Code Documentation

- **Docstrings**: Use Google-style docstrings for all public methods
- **Phase Numbers**: Include phase number in function docstrings
- **Math Notation**: Document mathematical formulas in comments

Example:
```python
def protect_liquid_warp(self, image_path: str, strength: int = 75):
    """
    Phase 17: Liquid Warp (Geometric Warping).
    
    Uses grid_sample to create smooth geometric distortions that
    disrupt identity while maintaining visual coherence.
    
    Args:
        image_path: Path to input image
        strength: Attack intensity (0-100)
        
    Returns:
        Protected PIL Image
        
    Math:
        displacement = tanh(θ) * flow_limit
        where θ is optimized via AdamW to minimize:
        L = cos_sim(SigLIP(warped), SigLIP(orig)) + λ*TV(displacement)
    """
```

### Updating README

When adding a public-facing feature:
1. Update method table in `README.md`
2. Add usage example if UI changes
3. Update Docker requirements if dependencies change

---

## 🛠️ Development Tips

### Fast Testing (No Container Rebuild)

Mount your local code during development:
```bash
docker run -it --gpus all \
  -v $(pwd)/src:/app/src \
  safespace:latest streamlit run src/app.py
```

### Debugging in Container

```bash
# Exec into running container
docker exec -it safespace-app bash

# Check logs
tail -f /app/logs/safespace.log
```

### Adding Dependencies

1. Update `requirements.txt`
2. Rebuild container: `docker build -t safespace:latest .`

---

## 📊 Performance Benchmarks

When contributing new methods, include benchmark results:

| Method | SSIM | PSNR | Bio-Sim | Time (s) | VRAM (GB) |
|--------|------|------|---------|----------|-----------|
| Your Method | 0.XX | XX dB | 0.XX | XX | XX |

Run benchmarks using:
```bash
python tests/benchmark_all_methods.py
```

---

## ❓ Questions?

- **Research Questions**: Open a discussion in Issues
- **Bug Reports**: Use GitHub Issues with `[BUG]` prefix
- **Feature Requests**: Use GitHub Issues with `[FEATURE]` prefix

---

## 📜 Code of Conduct

- Be respectful and collaborative
- Document your work thoroughly
- Test before submitting PRs
- Follow existing code style (PEP 8 for Python)

---

Thank you for contributing to SafeSpace! 🛡️
