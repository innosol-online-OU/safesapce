# Math Analysis of Project Invisible

## 1. Mathematical Models

### 1.1 Ghost Mesh Optimization (Phase 18)
The Ghost Mesh attack combines pixel-wise noise injection with geometric warping.

**Formula:**
$$x_{cloaked} = \text{GridSample}(x + \delta_{noise}, I + \delta_{warp})$$

**Loss Function (Hinge-Loss):**
$$L_{total} = W_{id} \cdot \max(0, \text{CosSim}(E(x_{warped}), E(x_{orig})) - \tau_{id}) + W_{lpips} \cdot \max(0, \text{LPIPS}(x_{warped}, x_{orig}) - \tau_{lpips}) + W_{tv} \cdot TV(\delta_{noise}) + W_{vert} \cdot |\delta_{warp}^y|$$

*   **Identity Loss ($W_{id}$):** Uses Cosine Similarity of SigLIP embeddings. The hinge $\tau_{id} = 0.25$ ensures the optimizer pushes until the identity is sufficiently broken, then stops to preserve quality.
*   **Visual Fidelity ($W_{lpips}$):** Penalizes perceptual degradation only if it exceeds $\tau_{lpips} = 0.065$.
*   **Regularization:** Total Variation (TV) ensures smoothness, and Vertical Penalty prevents unnatural eye movement.

### 1.2 Liquid Warp (Phase 17.9)
Focuses on geometric distortion without pixel noise, using a "T-Zone" anchor to freeze the face silhouette.

**Formula:**
$$x_{warped} = \text{GridSample}(x, I + M_{tzone} \cdot \tanh(\delta_{warp}) \cdot \epsilon)$$

*   **T-Zone Mask ($M_{tzone}$):** A Gaussian-feathered mask targeting eyes/nose/mouth ($1.0$) while fading to $0.0$ at the jawline.
*   **Constraints:** $\tanh$ bounds the displacement, scaled by $\epsilon$ (e.g., 0.03 normalized pixels).

### 1.3 Latent Diffusion Cloak
Standard PGD attack in the latent space of a VAE (Stable Diffusion).

**Formula:**
$$z_{adv} = z + \text{Clip}(\alpha \cdot \text{sign}(\nabla_z L), -\epsilon, \epsilon)$$

*   **Gradients:** Backpropagated through the VAE decoder (differentiable) to the latent $z$.
*   **Boost:** Gradients on the face are multiplied by `face_boost` (1.0 - 3.0) before the update.

## 2. Validation Metrics

*   **SSIM (Structural Similarity):**
    $$SSIM(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
    Used to measure low-level structural preservation (texture, edges).

*   **PSNR (Peak Signal-to-Noise Ratio):**
    $$PSNR = 20 \cdot \log_{10}(\frac{MAX_I}{\sqrt{MSE}})$$
    Standard measure of image quality.

*   **Bio-Sim (Identity Preservation):**
    Cosine similarity of InsightFace embeddings.
    $$Sim = \frac{A \cdot B}{||A|| \cdot ||B||}$$
    Target: $< 0.3$ (Unmatched).

## 3. Cheating / Errors Check

*   **Gradient Flow:** The use of `GridSample` (bilinear) and `VAE.decode` ensures end-to-end differentiability.
*   **Metric Validity:** SSIM and PSNR use standard implementations (`skimage` or `torch` equivalents). Bio-Sim uses a recognized industry-standard model (InsightFace/ArcFace).
*   **Optimization:** The use of Hinge Loss prevents "over-optimization" (making the image ugly just to get a lower score) by stopping once the threshold is met. This is a valid technique for adversarial attacks.
