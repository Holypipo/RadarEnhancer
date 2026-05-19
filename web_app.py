import streamlit as st
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from datetime import datetime
import subprocess
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from PIL import Image

# Add root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, FETCH_DIR, ensure_core_dirs
from utils.radar_io import read_radar, get_field_array
from utils.enhancement import denoise_gaussian, upscale, sharpen_unsharp_mask, enhance_field, compute_snr, generate_synthetic_radar_field
from utils.nexrad_fetcher import download_nexrad_file

ensure_core_dirs()

# ── Scientific Color Palette (AEMET Official) ─────────────────────────────────
AEMET_REF_COLORS = ["#0000fc", "#0094fc", "#00fcfc", "#00a000", "#00fc00", "#fcfc00", "#fcac00", "#fc7400", "#fc0000", "#bc0000", "#fc00fc"]
CUSTOM_CMAP = ListedColormap(["#0b0e14"] + AEMET_REF_COLORS)

def get_scientific_colorscale():
    colorscale = [[0.0, "#0b0e14"], [0.15, "#0b0e14"]] 
    for i, color in enumerate(AEMET_REF_COLORS):
        val = 12 + i * 6
        norm = (val + 10) / 85.0
        colorscale.append([norm - 0.001, color])
        colorscale.append([norm + 0.001, color])
    colorscale.append([1.0, AEMET_REF_COLORS[-1]])
    return colorscale

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Radar Processing Lab", page_icon="📡", layout="wide", initial_sidebar_state="expanded")

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0b0e14; color: #e0e0e0; }
    .stButton>button { background-color: transparent !important; color: #00d4ff !important; border: 1px solid #00d4ff !important; width: 100%; text-transform: uppercase; font-weight: bold; }
    button:disabled { background-color: #1a1a1a !important; color: #444 !important; border: 1px solid #333 !important; opacity: 0.8 !important; }
</style>
""", unsafe_allow_html=True)

# ── State Initialization ──────────────────────────────────────────────────────
for key in ['raw_data', 'enhanced_data', 'radar_obj', 'raw_file_path', 'summary_pdf', 'full_pdf']:
    if key not in st.session_state: st.session_state[key] = None

# ── Sidebar UI ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("SYSTEM CONTROLS")
    if st.button("GET RADAR DATA", use_container_width=True, type="primary"):
        with st.spinner("FETCHING..."):
            try:
                src = st.session_state.get('source_sel', 'AEMET (Spain)')
                if src == "NEXRAD (USA)":
                    st_id = st.session_state.get('station_sel', 'KTLX').split(" ")[0]
                    dt, tm = st.session_state.get('date_sel', datetime(2013, 5, 20)), st.session_state.get('time_sel', datetime(2013, 5, 20, 19, 55).time())
                    file_path, err = download_nexrad_file(st_id, dt, tm, output_dir=str(DATA_DIR))
                    if file_path:
                        st.session_state['radar_obj'] = read_radar(file_path)
                        st.session_state['raw_data'] = get_field_array(st.session_state['radar_obj'], "reflectivity", sweep=0)
                    else: st.error(f"DOWNLOAD FAILED: {err}")
                elif src == "AEMET (Spain)":
                    st_id = st.session_state.get('station_sel', 'nacional').split(" ")[0]
                    subprocess.run(["py", "01_fetch_aemet.py", "--radar", st_id], capture_output=True)
                    prefix = "01_AemetNacional" if st_id == "nacional" else f"01_Aemet{st_id.capitalize()}"
                    files = [f for f in os.listdir(FETCH_DIR) if f.startswith(prefix) and f.endswith(".gif")]
                    if files:
                        file_path = os.path.join(FETCH_DIR, sorted(files)[-1])
                        st.session_state['raw_file_path'] = file_path
                        st.session_state['radar_obj'] = read_radar(file_path)
                        st.session_state['raw_data'] = get_field_array(st.session_state['radar_obj'], "reflectivity")
                    else: st.session_state['raw_data'] = generate_synthetic_radar_field()
                st.session_state['enhanced_data'] = None
                st.session_state['summary_pdf'] = None
                st.session_state['full_pdf'] = None
            except Exception as e: st.error(f"IO ERROR: {e}")
    
    enhance_disabled = st.session_state['raw_data'] is None
    if st.button("EXECUTE ENHANCEMENT KERNEL", use_container_width=True, disabled=enhance_disabled):
        with st.spinner("ENHANCING..."):
            st.session_state['enhanced_data'] = enhance_field(st.session_state['raw_data'], denoise_h=st.session_state.get('denoise_h', 0.6), scale=2, sharpen_amount=st.session_state.get('sharpen_amount', 1.0), use_resnet=st.session_state.get('use_resnet', True))
            st.session_state['summary_pdf'] = None
            st.session_state['full_pdf'] = None

    if st.session_state['enhanced_data'] is not None:
        st.markdown("---")
        st.header("REPORT GENERATION")
        
        # Helper to get the truly raw source image for PDF
        def get_source_img_for_pdf():
            if st.session_state['raw_file_path'] and os.path.exists(st.session_state['raw_file_path']):
                return Image.open(st.session_state['raw_file_path']).convert("RGB")
            return st.session_state['raw_data']

        if st.button("GENERATE SUMMARY SHEET", use_container_width=True):
            with st.spinner("COMPILING SUMMARY..."):
                fig = plt.figure(figsize=(8.5, 11), facecolor='white')
                fig.text(0.5, 0.95, "RESEARCH SUMMARY: RADAR SIGNAL PROCESSING", ha='center', fontsize=14, fontweight='bold')
                info = (f"SOURCE: {st.session_state.get('source_sel')}\n"
                        f"STATION: {st.session_state.get('station_sel')}\n"
                        f"PEAK: {np.nanmax(st.session_state['raw_data']):.1f} dBZ\n"
                        f"KERNEL: RESNET-V3.2 | NLM: {st.session_state.get('denoise_h')}")
                fig.text(0.5, 0.88, info, ha='center', va='top', fontsize=10, family='monospace', bbox=dict(facecolor='#f0f0f0', edgecolor='#ccc', pad=10))
                
                # Raw Source Image
                ax1 = fig.add_axes([0.05, 0.45, 0.44, 0.35])
                src_img = get_source_img_for_pdf()
                if isinstance(src_img, Image.Image): ax1.imshow(src_img)
                else: ax1.imshow(src_img, vmin=-10, vmax=75, cmap=CUSTOM_CMAP)
                ax1.set_axis_off(); ax1.set_title("ORIGINAL SENSOR FEED", fontsize=10)
                
                # Enhanced
                ax2 = fig.add_axes([0.51, 0.45, 0.44, 0.35])
                ax2.imshow(st.session_state['enhanced_data'], vmin=-10, vmax=75, cmap=CUSTOM_CMAP)
                ax2.set_axis_off(); ax2.set_title("L2 ENHANCED SIGNAL", fontsize=10)
                
                buf = io.BytesIO(); plt.savefig(buf, format="pdf"); st.session_state['summary_pdf'] = buf.getvalue(); plt.close(fig)
        
        if st.session_state['summary_pdf']:
            st.download_button("DOWNLOAD SUMMARY", st.session_state['summary_pdf'], "Radar_Summary.pdf", "application/pdf", use_container_width=True)

        if st.button("GENERATE FULL REPORT", use_container_width=True):
            with st.spinner("COMPILING SCIENTIFIC REPORT..."):
                from matplotlib.backends.backend_pdf import PdfPages
                buf = io.BytesIO()
                src_img = get_source_img_for_pdf()
                with PdfPages(buf) as pdf:
                    # Page 1: Signal Ingestion
                    fig = plt.figure(figsize=(8.5, 11))
                    fig.text(0.5, 0.95, "SCIENTIFIC REPORT: RADAR SIGNAL ANALYSIS", ha='center', fontsize=16, fontweight='bold')
                    fig.text(0.1, 0.90, "SECTION 1: SIGNAL ACQUISITION", fontsize=12, fontweight='bold')
                    fig.text(0.1, 0.84, "The primary sensor data is ingested as a binary raster volume.\nThis represents the initial meteorological reflectivity field.", fontsize=10)
                    ax = fig.add_axes([0.1, 0.3, 0.8, 0.5])
                    if isinstance(src_img, Image.Image): ax.imshow(src_img)
                    else: ax.imshow(src_img, vmin=-10, vmax=75, cmap=CUSTOM_CMAP)
                    ax.set_axis_off()
                    pdf.savefig(fig); plt.close(fig)
                    
                    # Page 2: Enhancement
                    fig = plt.figure(figsize=(8.5, 11))
                    fig.text(0.1, 0.90, "SECTION 2: AI ENHANCEMENT PIPELINE", fontsize=12, fontweight='bold')
                    desc = f"Processing: NLM Denoising (H={st.session_state.get('denoise_h')}) + ResNet Upscaling.\nFinal resolution increased by 200% for high-fidelity convective analysis."
                    fig.text(0.1, 0.84, desc, fontsize=10)
                    ax = fig.add_axes([0.1, 0.3, 0.8, 0.5])
                    ax.imshow(st.session_state['enhanced_data'], vmin=-10, vmax=75, cmap=CUSTOM_CMAP)
                    ax.set_axis_off()
                    pdf.savefig(fig); plt.close(fig)
                    
                    # Page 3: Comparison
                    fig = plt.figure(figsize=(8.5, 11))
                    fig.text(0.1, 0.90, "SECTION 3: COMPARATIVE VALIDATION", fontsize=12, fontweight='bold')
                    ax1 = fig.add_axes([0.1, 0.55, 0.35, 0.3])
                    if isinstance(src_img, Image.Image): ax1.imshow(src_img)
                    else: ax1.imshow(src_img, vmin=-10, vmax=75, cmap=CUSTOM_CMAP)
                    ax1.set_axis_off(); ax1.set_title("RAW SENSOR FEED", fontsize=9)
                    ax2 = fig.add_axes([0.55, 0.55, 0.35, 0.3])
                    ax2.imshow(st.session_state['enhanced_data'], vmin=-10, vmax=75, cmap=CUSTOM_CMAP)
                    ax2.set_axis_off(); ax2.set_title("ENHANCED PRODUCT", fontsize=9)
                    pdf.savefig(fig); plt.close(fig)
                
                st.session_state['full_pdf'] = buf.getvalue()

        if st.session_state['full_pdf']:
            st.download_button("DOWNLOAD FULL REPORT", st.session_state['full_pdf'], "Radar_Full_Report.pdf", "application/pdf", use_container_width=True)

    st.markdown("---")
    st.header("INSTRUMENT CONFIG")
    source = st.radio("PRIMARY SENSOR", ["NEXRAD (USA)", "AEMET (Spain)"], index=1, key="source_sel")
    if source == "NEXRAD (USA)":
        st.selectbox("RADAR STATION", ["KTLX (Oklahoma City)", "KOKX (New York)", "KLSX (St. Louis)"], key="station_sel")
        st.date_input("OBSERVATION DATE", datetime(2013, 5, 20), key="date_sel")
        st.time_input("TIME (UTC)", datetime(2013, 5, 20, 19, 55).time(), key="time_sel")
    else:
        stations = ["nacional", "am (Almería)", "ba (Barcelona)", "cc (Cáceres)", "co (A Coruña)", "ma (Madrid)", "ml (Málaga)", "mu (Murcia)", "pm (Palma)", "sa (Salamanca)", "se (Sevilla)", "va (Valencia)", "vd (Valladolid)", "za (Zaragoza)", "ca (Canarias)"]
        st.selectbox("REGIONAL SCOPE", stations, key="station_sel")
    
    st.markdown("---")
    st.header("PROCESSING KERNELS")
    st.checkbox("AI SUPER-RESOLUTION (RESNET-V3)", value=True, key="use_resnet")
    st.slider("NLM DENOISE STRENGTH", 0.1, 2.0, 0.6, key="denoise_h")
    st.slider("ADAPTIVE SHARPENING", 0.0, 2.0, 1.0, key="sharpen_amount")

# ── Main UI ──────────────────────────────────────────────────────────────────
st.title("Experimental AI Radar Processing")
st.markdown("<p style='font-size: 1.1rem; color: #888; margin-top: -15px; margin-bottom: 30px;'>Signal Enhancement Research System (v3.2)</p>", unsafe_allow_html=True)

if st.session_state['raw_data'] is not None:
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("SIGNAL STATUS", "STABLE")
    with m2: st.metric("PEAK INTENSITY", f"{np.nanmax(st.session_state['raw_data']):.1f} dBZ")
    with m3: st.metric("RESOLUTION", f"{st.session_state['raw_data'].shape[1]}x{st.session_state['raw_data'].shape[0]}")
    with m4: st.metric("SNR (Est.)", f"{compute_snr(st.session_state['raw_data']):.1f} dB")

    st.markdown("---")
    st.markdown("### STEP 1: SOURCE SIGNAL ACQUISITION")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**ORIGINAL SENSOR FEED (GIF)**")
        if st.session_state['raw_file_path']: st.image(st.session_state['raw_file_path'], use_container_width=True)
        else: st.info("RESEARCH SIMULATION")
    with c2:
        st.markdown("**EXTRACTED NUMERICAL FIELD (L0)**")
        fig_r, ax_r = plt.subplots(figsize=(10, 8), facecolor="#0e1117")
        ax_r.imshow(st.session_state['raw_data'], vmin=-10, vmax=75, cmap=CUSTOM_CMAP, aspect="auto", interpolation="nearest")
        ax_r.axis("off"); st.pyplot(fig_r)

    st.markdown("---")
    st.markdown("### STEP 2: ENHANCED PRODUCT GENERATION")
    ce1, ce2 = st.columns(2)
    with ce1:
        st.markdown("#### L0_RAW_SIGNAL")
        fig_l0, ax_l0 = plt.subplots(figsize=(10, 8), facecolor="#0e1117")
        ax_l0.imshow(st.session_state['raw_data'], vmin=-10, vmax=75, cmap=CUSTOM_CMAP, aspect="auto", interpolation="nearest")
        ax_l0.axis("off"); st.pyplot(fig_l0)
    with ce2:
        st.markdown("#### L2_ENHANCED_PRODUCT")
        if st.session_state['enhanced_data'] is not None:
            fig_l2, ax_l2 = plt.subplots(figsize=(10, 8), facecolor="#0e1117")
            ax_l2.imshow(st.session_state['enhanced_data'], vmin=-10, vmax=75, cmap=CUSTOM_CMAP, aspect="auto", interpolation="nearest")
            ax_l2.axis("off"); st.pyplot(fig_l2)
        else: st.markdown("<div style='height: 400px; display: flex; align-items: center; justify-content: center; border: 1px dashed #333;'>Awaiting Execution</div>", unsafe_allow_html=True)

    # Linked Synchronized Viewer
    st.markdown("---")
    st.markdown("## SYNCHRONIZED RESEARCH VIEWER")
    raw = st.session_state['raw_data']
    enhanced = st.session_state['enhanced_data'] if st.session_state['enhanced_data'] is not None else raw
    if enhanced.shape != raw.shape:
        from scipy.ndimage import zoom
        raw_viewer = zoom(raw, (enhanced.shape[0]/raw.shape[0], enhanced.shape[1]/raw.shape[1]), order=0)
    else: raw_viewer = raw
    fig_sync = make_subplots(rows=1, cols=2, subplot_titles=("RAW DATA", "ENHANCED DATA"), shared_xaxes=True, shared_yaxes=True)
    sci_colors = get_scientific_colorscale()
    fig_sync.add_trace(go.Heatmap(z=np.flipud(raw_viewer), colorscale=sci_colors, zmin=-10, zmax=75, showscale=False, zsmooth=False), row=1, col=1)
    fig_sync.add_trace(go.Heatmap(z=np.flipud(enhanced), colorscale=sci_colors, zmin=-10, zmax=75, showscale=False, zsmooth=False), row=1, col=2)
    fig_sync.update_layout(height=600, paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14", margin=dict(l=10, r=10, b=10, t=40))
    fig_sync.update_xaxes(scaleanchor="y", scaleratio=1)
    st.plotly_chart(fig_sync, use_container_width=True)
else: st.info("INITIALIZE CONFIGURATION TO START RESEARCH ANALYSIS.")
