# SafeSpace Research

This folder contains research experiments, model training, and notebooks for developing novel adversarial attacks.

## Structure

- **`notebooks/`**: Jupyter notebooks for experiments and ablation studies
- **`training/`**: Model training scripts (surrogate models, critics)
- **`experiments/`**: One-off experiments and prototypes

## Getting Started

### Running Notebooks

```bash
# Install Jupyter in container
docker exec -it safespace-app pip install jupyter

# Start Jupyter server
docker exec -it safespace-app jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root

# Access: http://localhost:8888
```

### Example Notebook Structure

```python
# 1. Setup
from invisible_core.attacks.latent_cloak import LatentCloak
import numpy as np
import matplotlib.pyplot as plt

# 2. Load model
cloak = LatentCloak()

# 3. Run experiment
results = []
for strength in range(10, 100, 10):
    result = cloak.protect_liquid_warp("test.png", strength=strength)
    # Evaluate metrics
    results.append(metrics)

# 4. Visualize
plt.plot(strengths, ssim_scores)
plt.show()
```

## Contributing Research

See [../CONTRIBUTING.md](../CONTRIBUTING.md) for:
- Adding new attack methods
- Experiment documentation standards
- Branch strategy for research work
