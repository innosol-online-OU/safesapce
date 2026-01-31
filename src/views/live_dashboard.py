
import streamlit as st
import time
import matplotlib.pyplot as plt
import numpy as np

def render_live_dashboard(manager):
    """
    Renders the Live Optimization Dashboard.
    Control flow is managed via manager methods.
    """
    st.divider()
    st.markdown("### 👻 Ghost-Mesh Operations Center")
    
    col_img, col_metrics = st.columns([1, 1])
    
    # 1. Visualizations (Images)
    with col_img:
        current_img = manager.get_current_image()
        orig_img = manager.get_original_image()
        components = manager.get_components() # Get noise/warp
        
        if current_img:
            # Display Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["👁️ View", "🔥 Total", "🌫️ Noise", "🌀 Warp"])
            
            with tab1:
                st.caption("**Protected Output**: The final image tailored to deceive AI while maintaining visual quality.")
                st.image(current_img, caption="Protected Output", width="stretch")
            
            with tab2:
                st.caption("**Total Perturbation**: Combined magnitude of Noise and Warping. Shows where the image was modified most.")
                if orig_img:
                    curr_np = np.array(current_img).astype(float)
                    orig_np = np.array(orig_img).astype(float)
                    diff = np.abs(curr_np - orig_np).mean(axis=2)
                    
                    fig_hm, ax_hm = plt.subplots(figsize=(5, 5), facecolor='#0e0e12')
                    ax_hm.imshow(diff, cmap='turbo', vmin=0, vmax=50)
                    ax_hm.axis('off')
                    st.pyplot(fig_hm)
                    plt.close(fig_hm)
            
            with tab3:
                st.caption("**Noise Layer (Resonant Ghost)**: High-frequency pixel noise targeting texture analysis.")
                if 'noise' in components:
                    fig_n, ax_n = plt.subplots(figsize=(5, 5), facecolor='#0e0e12')
                    ax_n.imshow(components['noise'], cmap='magma')
                    ax_n.axis('off')
                    st.pyplot(fig_n)
                    plt.close(fig_n)
                else:
                    st.caption("No noise data yet.")
                    
            with tab4:
                st.caption("**Warp Field (Liquid Warp)**: Geometric distortions shifting facial landmarks spatially.")
                if 'warp' in components:
                    fig_w, ax_w = plt.subplots(figsize=(5, 5), facecolor='#0e0e12')
                    ax_w.imshow(components['warp'], cmap='viridis')
                    ax_w.axis('off')
                    st.pyplot(fig_w)
                    plt.close(fig_w)
                else:
                    st.caption("No warp data yet.")

        else:
            st.info("Initializing Tensor Field...")
            
    # 2. Metrics & Controls
    with col_metrics:
        # ... Controls logic remains ...
        status = manager.get_status()
        progress = manager.get_progress()
        st.progress(progress, text=f"Status: {status} ({int(progress*100)}%)")
        
        c1, c2, c3, c4 = st.columns(4)
        # Re-implement buttons...
        if status == "RUNNING":
            if c1.button("⏸️ PAUSE"): manager.pause(); st.rerun()
            if c4.button("🛑 STOP"): manager.stop(); st.rerun()
        elif status == "PAUSED":
            if c1.button("▶️ RESUME"): manager.resume(); st.rerun()
            if c2.button("👟 STEP"): manager.step(1); st.rerun()
            if c3.button("🔍 CHECK"): manager.trigger_validation(); st.rerun()
            if c4.button("🛑 STOP"): manager.stop(); st.rerun()
        elif status in ["COMPLETED", "STOPPED"]:
            st.success("Sequence Complete.")
            if st.button("🔄 NEW RUN"): 
                st.session_state['live_session_active'] = False
                st.rerun()
        
        # Download Button (Always available if image exists)
        img_bytes = manager.get_current_image_bytes()
        if img_bytes:
             st.download_button("💾 Download Image", data=img_bytes, file_name="safespace_protected.png", mime="image/png")


            # 3. Vector Trajectory Plot (Identity Similarity)
        metrics = manager.get_metrics()
        logs = manager.get_logs()
        
        if metrics and len(metrics['step']) > 0:
            with st.expander("📉 Identity Vector Visualization", expanded=True):
                steps = metrics['step']
                # Use Cossim if available, else fallback to Loss
                if 'cossim' in metrics and len(metrics['cossim']) > 0:
                    y_data = metrics['cossim']
                    y_label = "Similarity Score (Lower is Better)"
                    safe_thresh = 0.25
                else:
                    y_data = metrics['identity_loss']
                    y_label = "Identity Loss"
                    safe_thresh = 0.0

                fig, ax = plt.subplots(figsize=(6, 3), facecolor='#0e0e12')
                ax.set_facecolor('#0e0e12')
                
                # Plot Data
                # Color: Red if high (unsafe), fading to Green if low? 
                # Just simple Red line.
                ax.plot(steps, y_data, color='#f43f5e', linewidth=2, label=y_label)
                
                # Plot Threshold
                ax.axhline(y=safe_thresh, color='#10b981', linestyle='--', linewidth=1, label='Safe Zone (<0.25)')
                
                # Fill Safe Zone
                ax.fill_between(steps, 0, safe_thresh, color='#10b981', alpha=0.1)
                
                ax.set_title(f"Target: Break Identity (Score < {safe_thresh})", color='white', fontsize=10)
                ax.tick_params(colors='#9ca3af', labelsize=8)
                ax.set_ylim(-0.05, 1.05) if 'cossim' in metrics else None
                ax.grid(color='#374151', linestyle=':', linewidth=0.5)
                # Spines
                for s in ax.spines.values(): s.set_visible(False)
                ax.spines['left'].set_visible(True); ax.spines['left'].set_color('#374151')
                ax.spines['bottom'].set_visible(True); ax.spines['bottom'].set_color('#374151')
                
                st.pyplot(fig)
                plt.close(fig)

            # 4. Validation Status
            last_val = manager.get_last_validation()
            if last_val:
                if "Protected" in last_val:
                     st.success(f"🛡️ **Status**: {last_val}")
                else:
                     st.warning(f"⚠️ **Status**: {last_val}")

            # 5. System Logs
            with st.expander("💻 System Terminal", expanded=True):
                log_txt = "\n".join(logs[-20:]) if logs else "Waiting for logs..."
                st.code(log_txt, language="text")


    # 4. Auto-Advance Logic
    if status == "RUNNING":
        manager.step(steps_per_chunk=5)
        time.sleep(0.05) # Tiny yield to UI
        st.rerun()
    
    # 5. Final Report (On Completion)
    if status in ["COMPLETED", "STOPPED"] and not st.session_state.get('report_shown'):
         # We could instantiate the Qwen check here!
         # For now, just show the final best image
         pass

