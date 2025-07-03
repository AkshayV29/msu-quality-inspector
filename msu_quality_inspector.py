import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
from skimage.metrics import structural_similarity as ssim
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

# Configure page
st.set_page_config(
    page_title="MSU Quality Inspector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
    .pass-status {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .fail-status {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .warning-status {
        color: #ffc107;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'inspection_history' not in st.session_state:
    st.session_state.inspection_history = []
if 'golden_image' not in st.session_state:
    st.session_state.golden_image = None

def preprocess_image(image, target_size=(800, 600)):
    """Preprocess image for comparison"""
    # Convert PIL to numpy array
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Resize to standard size
    resized = cv2.resize(gray, target_size)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    
    return blurred

def calculate_similarity_metrics(golden_img, test_img):
    """Calculate various similarity metrics"""
    # Structural Similarity Index
    ssim_score = ssim(golden_img, test_img)
    
    # Mean Squared Error
    mse = np.mean((golden_img.astype(float) - test_img.astype(float)) ** 2)
    
    # Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Normalized Cross-Correlation
    ncc = cv2.matchTemplate(golden_img, test_img, cv2.TM_CCORR_NORMED)[0, 0]
    
    return {
        'ssim': ssim_score,
        'mse': mse,
        'psnr': psnr,
        'ncc': ncc
    }

def detect_differences(golden_img, test_img):
    """Detect and highlight differences between images"""
    # Calculate absolute difference
    diff = cv2.absdiff(golden_img, test_img)
    
    # Apply threshold to get binary difference image
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Find contours of differences
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create difference overlay
    diff_overlay = cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGB)
    
    # Draw rectangles around differences
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Filter small noise
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(diff_overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return diff_overlay, len(contours)

def determine_quality_status(metrics, difference_count):
    """Determine quality status based on metrics"""
    ssim_threshold = 0.85
    mse_threshold = 1000
    diff_count_threshold = 5
    
    if (metrics['ssim'] >= ssim_threshold and 
        metrics['mse'] <= mse_threshold and 
        difference_count <= diff_count_threshold):
        return "PASS", "✅"
    elif metrics['ssim'] >= 0.7:
        return "WARNING", "⚠️"
    else:
        return "FAIL", "❌"

def create_metrics_chart(metrics):
    """Create a radar chart for metrics visualization"""
    categories = ['SSIM', 'PSNR (norm)', 'NCC', 'MSE (inv norm)']
    
    # Normalize metrics for radar chart
    values = [
        metrics['ssim'],
        min(metrics['psnr'] / 50, 1.0),  # Normalize PSNR
        metrics['ncc'],
        1 - min(metrics['mse'] / 2000, 1.0)  # Invert and normalize MSE
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Quality Metrics',
        line_color='#2E86AB'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        height=300,
        title="Quality Metrics Overview"
    )
    
    return fig

# Main App
st.markdown('<div class="main-header">🔍 MSU Quality Inspector</div>', unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Golden image upload
    st.subheader("Golden MSU Reference")
    golden_file = st.file_uploader(
        "Upload Golden MSU Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload the reference image for comparison"
    )
    
    if golden_file is not None:
        st.session_state.golden_image = Image.open(golden_file)
        st.success("Golden image loaded!")
        st.image(st.session_state.golden_image, caption="Golden MSU", width=200)
    
    # Quality thresholds
    st.subheader("Quality Thresholds")
    ssim_threshold = st.slider("SSIM Threshold", 0.0, 1.0, 0.85, 0.05)
    mse_threshold = st.slider("MSE Threshold", 0, 2000, 1000, 50)
    diff_threshold = st.slider("Max Differences", 0, 20, 5, 1)
    
    # Inspection history
    st.subheader("📊 Inspection Stats")
    if st.session_state.inspection_history:
        df = pd.DataFrame(st.session_state.inspection_history)
        total_inspections = len(df)
        pass_count = len(df[df['status'] == 'PASS'])
        fail_count = len(df[df['status'] == 'FAIL'])
        warning_count = len(df[df['status'] == 'WARNING'])
        
        st.metric("Total Inspections", total_inspections)
        st.metric("Pass Rate", f"{(pass_count/total_inspections)*100:.1f}%")
        
        # Status distribution
        fig_pie = px.pie(
            values=[pass_count, warning_count, fail_count],
            names=['PASS', 'WARNING', 'FAIL'],
            color_discrete_map={
                'PASS': '#28a745',
                'WARNING': '#ffc107',
                'FAIL': '#dc3545'
            }
        )
        fig_pie.update_layout(height=200, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

# Main content area
if st.session_state.golden_image is None:
    st.warning("⚠️ Please upload a golden MSU reference image in the sidebar first.")
    st.stop()

# Test image upload
st.subheader("📸 Upload MSU for Inspection")
test_file = st.file_uploader(
    "Upload Test MSU Image",
    type=['jpg', 'jpeg', 'png'],
    help="Upload the MSU image to be inspected"
)

if test_file is not None:
    test_image = Image.open(test_file)
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🥇 Golden MSU")
        st.image(st.session_state.golden_image, caption="Reference Image", use_column_width=True)
    
    with col2:
        st.subheader("🔍 Test MSU")
        st.image(test_image, caption="Test Image", use_column_width=True)
    
    # Process images
    with st.spinner("Processing images and calculating metrics..."):
        # Preprocess images
        golden_processed = preprocess_image(st.session_state.golden_image)
        test_processed = preprocess_image(test_image)
        
        # Calculate metrics
        metrics = calculate_similarity_metrics(golden_processed, test_processed)
        
        # Detect differences
        diff_overlay, difference_count = detect_differences(golden_processed, test_processed)
        
        # Determine quality status
        status, status_icon = determine_quality_status(metrics, difference_count)
    
    # Display results
    st.subheader("📊 Inspection Results")
    
    # Status display
    status_col1, status_col2, status_col3 = st.columns([2, 1, 1])
    
    with status_col1:
        if status == "PASS":
            st.markdown(f'<div class="pass-status">{status_icon} {status} - Quality Approved</div>', unsafe_allow_html=True)
        elif status == "WARNING":
            st.markdown(f'<div class="warning-status">{status_icon} {status} - Review Required</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="fail-status">{status_icon} {status} - Quality Rejected</div>', unsafe_allow_html=True)
    
    with status_col2:
        st.metric("Differences Found", difference_count)
    
    with status_col3:
        st.metric("SSIM Score", f"{metrics['ssim']:.3f}")
    
    # Detailed metrics
    st.subheader("📈 Detailed Metrics")
    
    metrics_col1, metrics_col2 = st.columns([1, 1])
    
    with metrics_col1:
        st.metric("Structural Similarity (SSIM)", f"{metrics['ssim']:.4f}", 
                 help="Higher is better (0-1)")
        st.metric("Mean Squared Error (MSE)", f"{metrics['mse']:.2f}", 
                 help="Lower is better")
        st.metric("Peak SNR (PSNR)", f"{metrics['psnr']:.2f} dB", 
                 help="Higher is better")
        st.metric("Normalized Cross-Correlation", f"{metrics['ncc']:.4f}", 
                 help="Higher is better (0-1)")
    
    with metrics_col2:
        # Radar chart
        fig_radar = create_metrics_chart(metrics)
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Difference visualization
    st.subheader("🔍 Difference Analysis")
    
    diff_col1, diff_col2 = st.columns(2)
    
    with diff_col1:
        st.subheader("Difference Heatmap")
        diff_gray = cv2.absdiff(golden_processed, test_processed)
        fig_heatmap = px.imshow(diff_gray, color_continuous_scale='hot', 
                               title="Intensity Differences")
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with diff_col2:
        st.subheader("Highlighted Differences")
        st.image(diff_overlay, caption="Red boxes show detected differences", use_column_width=True)
    
    # Save inspection record
    inspection_record = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'status': status,
        'ssim': metrics['ssim'],
        'mse': metrics['mse'],
        'psnr': metrics['psnr'],
        'ncc': metrics['ncc'],
        'differences': difference_count,
        'filename': test_file.name
    }
    
    # Add to history button
    if st.button("📝 Save Inspection Record", type="primary"):
        st.session_state.inspection_history.append(inspection_record)
        st.success("Inspection record saved!")
        st.rerun()

# Inspection history table
if st.session_state.inspection_history:
    st.subheader("📋 Inspection History")
    
    df = pd.DataFrame(st.session_state.inspection_history)
    df = df.sort_values('timestamp', ascending=False)
    
    # Color code the status
    def color_status(val):
        if val == 'PASS':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'WARNING':
            return 'background-color: #fff3cd; color: #856404'
        else:
            return 'background-color: #f8d7da; color: #721c24'
    
    styled_df = df.style.applymap(color_status, subset=['status'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Export history
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Inspection History",
        data=csv,
        file_name=f"msu_inspection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Clear history
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.inspection_history = []
        st.success("History cleared!")
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "MSU Quality Inspector v1.0 | Built with Streamlit | "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    "</div>",
    unsafe_allow_html=True
)