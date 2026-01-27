# Phase 18 UI Plan: Ghost-Mesh Protocol (Revised)

## 🎯 Objective
Combine the controls of **Phase 16 (Resonant Ghost)** and **Phase 17 (Liquid Warp)** into a unified interface for **Phase 18 (Ghost-Mesh)**, with enhanced feedback for complex operations.

## 1. Dropdown Integration
*   **Action**: Add "Ghost-Mesh (Phase 18)" to the `target_profile_options` dictionary.
*   **Key**: `ghost_mesh`
*   **Label**: `Ghost-Mesh (Phase 18: Hybrid)`

## 2. Combined Controls (The "Ghost-Mesh" Expander)

When `Ghost-Mesh` is selected, display a dedicated expander with the following controls:

### A. Attack Dynamics (The "Engine")
| UI Element | Label | Default | Range | Math Mapping |
| :--- | :--- | :--- | :--- | :--- |
| `st.slider` | **Attack Intensity** | 75 | 1-100 | Controls `epsilon` and `flow_limit`. |
| `st.slider` | **Warp/Noise Balance** | 0.5 | 0.0-1.0 | Interpolates between Warp-heavy (0.0) and Noise-heavy (1.0). |

### B. Structural Controls (From Phase 17)
| UI Element | Label | Default | Range | Math Mapping |
| :--- | :--- | :--- | :--- | :--- |
| `st.select_slider` | **Grid Resolution** | 12 | [12, 16, 24, 32] | Low (12) = global shifts. High (32) = local distortion. |
| `st.slider` | **T-Zone Anchoring** | 0.8 | 0.0-1.0 | 1.0 = Freeze jawline. 0.0 = Full warp. |

### C. Visual Stealth (From Phase 16)
| UI Element | Label | Default | Range | Default |
| :--- | :--- | :--- | :--- | :--- |
| `st.slider` | **Grain Control (TV)** | 50 | 1-100 | Smoothness. |
| `st.checkbox` | **Ghost Masking (JND)** | True | Bool | Texture-only noise. |

## 3. Visualization Updates (New Requests)

### A. "Panic Mode" Alert (Retry Loop)
*   **Logic**: If Qwen-VL detects identity after optimization, the engine triggers a retry.
*   **UI**: Show a warning status in the metrics area.
    *   **Normal**: `[INFO] Optimization Complete.`
    *   **Panic**: `[WARN] Identity Detected! Retrying with higher noise... (Attempt 2/3)`
    *   **Fail**: `[FAIL] Defense Compromised. Try increasing intensity.`

### B. "Live Mesh" Toggle
*   **Action**: Add a toggle to visualize the geometric transformation.
*   **UI Element**: `st.toggle("Visualize Mesh Distortion")`
*   **Implementation**:
    *   If **ON**: Display the *displacement grid* (vector field or warped grid lines) instead of the final image in the `Warp Map` slot.
    *   If **OFF**: Show the standard heatmap or difference map.

## 4. Implementation Snippet (`app.py`)

```python
elif is_ghost_mesh:
    st.info("👻🕸️ **Phase 18 (Ghost-Mesh)**\n\nCoupled Warp + Noise Optimization with Hinge-Loss Constraints.")
    
    with st.expander("👻🕸️ Ghost-Mesh Controls", expanded=True):
        # ... (Previous controls)
        
        # New Visualization Toggle
        st.divider()
        gm_show_mesh = st.toggle("Visualize Mesh Distortion", value=False, help="Show the geometric warp grid instead of the heatmap.")
        st.session_state['gm_show_mesh'] = gm_show_mesh

    # ... During Metrics Display ...
    if metrics.get('panic_mode', False):
        st.error(f"⚠️ **PANIC MODE ACTIVE**: Identity persisted after optimization. Retrying... (Attempt {metrics.get('attempt')})")
```
