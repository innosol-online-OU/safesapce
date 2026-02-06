# SafeSpace: Adversarial Defense Against AI Surveillance

Protect your identity from unauthorized AI training and biometric harvesting.

---

## Quick Start

```bash
# Build
docker build -t safespace .

# Run
docker run -d --gpus all -p 8501:8501 -v $(pwd)/uploads:/app/uploads safespace

# Access
http://localhost:8501
```

**Requirements:** Docker + NVIDIA GPU (8GB+ VRAM recommended)

---

## Methods

| Method | Type | Use Case |
|--------|------|----------|
| **Liquid Warp** | Geometric | Invisible distortion, highest stealth |
| **Resonant Ghost** | Pixel | Research/experimentation |
| **Frontier Lite** | Anti-Seg | Fast protection (<1GB VRAM) |
| **General** | Legacy | SD/Midjourney |

---

## Usage

1. Upload image → Select method → Adjust strength → Activate Defense
2. Download protected image

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Adding new attacks
- Research workflow (notebooks/training)
- Branch strategy (`main`/`dev`/`research`)
- Testing protocols

---

## Structure

```
src/          # Production code
research/     # Notebooks, model training
tests/        # Verification scripts
docs/         # Internal (gitignored)
```

---

## License

[TBD]

**Project Invisible © 2026**

