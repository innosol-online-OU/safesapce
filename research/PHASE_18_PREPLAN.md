# Phase 18 Pre-plan: Hybrid Architecture Status Report

## 1. Repository Rules (Strict Compliance)
*   **Branches**:
    *   `main`: Stable, production-ready. **NO direct commits**.
    *   `dev`: Integration branch.
    *   `research`: Daily driver for experiments and new phases.
*   **Commit Protocol**:
    *   Format: `type(scope): subject` (e.g., `feat(phase-18): initial setup`).
    *   **Agent Check**: verify branch before push.
*   **Documentation**:
    *   Update `README.md` and method tables when adding public features.
    *   Benchmarks required for new methods.

## 2. Phase 16: Resonant Ghost (Pixel-Level Adversarial Noise)
*   **Status**: Implemented & Verified.
*   **Method Name**: `protect_phantom`
*   **Location**: `src/core/protocols/latent_cloak.py`
*   **Core Mechanism**:
    *   **MI-FGSM** (Momentum Iterative Fast Gradient Sign Method).
    *   **DIM** (Input Diversity Method) for scale invariance.
    *   **SigLIP Critic**: Optimizes against ViT-SO400M embeddings.
*   **Key Features**:
    *   **JND Clamping**: Restricts noise to textured areas (hair/edges) using Sobel gradients.
    *   **Targeting**: Optional neural targeting to focus attacks on faces.
*   **Current Metrics**:
    *   Effective at high strength but can introduce visible "grain".

## 3. Phase 17: Liquid Warp (Geometric Distortion)
*   **Status**: Implemented & Verified (V2).
*   **Method Name**: `protect_liquid_warp_v2`
*   **Location**: `src/core/protocols/latent_cloak.py`
*   **Core Mechanism**:
    *   **Grid Sample**: Optimizes a low-res (12x12 or 16x16) displacement grid.
    *   **Focal Length Attack**: Simulates wide-angle lens distortion (horizontal expansion).
    *   **T-Zone Anchoring**: Warps internal features while freezing the silhouette (jawline) to prevent artifacts.
*   **Key Features**:
    *   **Silent**: No pixel noise, only geometric shifts.
    *   **Multi-Scale Loss**: Optimizes scale invariance (224px & 384px).
    *   **Vertical Penalty**: Prevents unnatural vertical shifting ("droopy eyes").
*   **Current Metrics**:
    *   High visual stealth (looks like a "bad selfie" rather than an attack).
    *   Breaks alignment-based recognition.

## 4. Phase 18: Proposed Hybrid Architecture
*   **Objective**: Combine the **structural accumulation** of Liquid Warp with the **feature disruption** of Resonant Ghost.
*   **Hypothesis**:
    *   Geometric warping misaligns the face, effectively "lowering the guard" of the recognition model.
    *   Pixel noise then has an easier job pushing the embedding away, requiring less `epsilon` (visibility).
*   **Proposed Flow**:
    1.  **Stage 1 (Liquid)**: Apply `protect_liquid_warp_v2` to structurally shift features (T-Zone expansion).
    2.  **Stage 2 (Ghost)**: Apply `protect_phantom` on the *warped* image, using the warped coordinates as the new baseline.
*   **Challenges**:
    1.  **Artifact Amp**: Ensuring the noise doesn't highlight the warp.
    2.  **Runtime**: Managing total runtime (sum of two optimizations).
