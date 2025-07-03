import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
from skimage.metrics import structural_similarity as ssim
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import json
import zipfile
from io import BytesIO

# Configure page
st.set_page_config(
    page_title="MSU Quality Inspector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="auto"
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
    .admin-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #dc3545;
        text-align: center;
        margin-bottom: 2rem;
    }
    .user-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #28a745;
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
    .login-container {
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background: white;
    }
    .role-button {
        width: 100%;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s;
    }
    .admin-button {
        background: #dc3545;
        color: white;
    }
    .user-button {
        background: #28a745;
        color: white;
    }
    .admin-button:hover, .user-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Mobile optimizations */
    @media (max-width: 768px) {
        .main-header, .admin-header, .user-header {
            font-size: 2rem;
        }
        .stButton > button {
            width: 100%;
            margin: 0.5rem 0;
        }
        .metric-card {
            padding: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'golden_image_data' not in st.session_state:
        st.session_state.golden_image_data = None
    if 'golden_image_info' not in st.session_state:
        st.session_state.golden_image_info = {}
    if 'inspection_history' not in st.session_state:
        st.session_state.inspection_history = []
    if 'quality_thresholds' not in st.session_state:
        st.session_state.quality_thresholds = {
            'ssim': 0.85,
            'mse': 1000,
            'max_differences': 5
        }

def save_golden_image(image, filename, admin_notes=""):
    """Save golden image to session state with metadata"""
    # Convert PIL image to base64 for storage
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    st.session_state.golden_image_data = img_str
    st.session_state.golden_image_info = {
        'filename': filename,
        'upload_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'admin_notes': admin_notes,
        'image_size': image.size
    }

def load_golden_image():
    """Load golden image from session state"""
    if st.session_state.golden_image_data:
        img_data = base64.b64decode(st.session_state.golden_image_data)
        return Image.open(BytesIO(img_data))
    return None

def preprocess_image(image, target_size=(800, 600)):
    """Preprocess image for comparison"""
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    resized = cv2.resize(gray, target_size)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    
    return blurred

def calculate_similarity_metrics(golden_img, test_img):
    """Calculate various similarity metrics"""
    ssim_score = ssim(golden_img, test_img)
    mse = np.mean((golden_img.astype(float) - test_img.astype(float)) ** 2)
    
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    ncc = cv2.matchTemplate(golden_img, test_img, cv2.TM_CCORR_NORMED)[0, 0]
    
    return {
        'ssim': ssim_score,
        'mse': mse,
        'psnr': psnr,
        'ncc': ncc
    }

def detect_differences(golden_img, test_img):
    """Detect and highlight differences between images"""
    diff = cv2.absdiff(golden_img, test_img)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    diff_overlay = cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGB)
    
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(diff_overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return diff_overlay, len(contours)

def determine_quality_status(metrics, difference_count, thresholds):
    """Determine quality status based on metrics"""
    if (metrics['ssim'] >= thresholds['ssim'] and 
        metrics['mse'] <= thresholds['mse'] and 
        difference_count <= thresholds['max_differences']):
        return "PASS", "✅"
    elif metrics['ssim'] >= 0.7:
        return "WARNING", "⚠️"
    else:
        return "FAIL", "❌"

def create_analytics_charts(df):
    """Create analytics charts from inspection data"""
    if df.empty:
        return None, None, None
    
    # Daily inspection count
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_counts = df.groupby('date').size().reset_index(name='count')
    
    fig_daily = px.line(daily_counts, x='date', y='count', 
                       title='Daily Inspections Count',
                       labels={'date': 'Date', 'count': 'Number of Inspections'})
    fig_daily.update_traces(line_color='#2E86AB', line_width=3)
    
    # Pass/Fail distribution
    status_counts = df['status'].value_counts()
    fig_status = px.pie(values=status_counts.values, names=status_counts.index,
                       title='Quality Status Distribution',
                       color_discrete_map={
                           'PASS': '#28a745',
                           'WARNING': '#ffc107',
                           'FAIL': '#dc3545'
                       })
    
    # Quality metrics over time
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Scatter(x=df['timestamp'], y=df['ssim'],
                                   mode='lines+markers', name='SSIM',
                                   line=dict(color='#2E86AB')))
    fig_metrics.update_layout(title='SSIM Quality Trend Over Time',
                            xaxis_title='Time',
                            yaxis_title='SSIM Score')
    
    return fig_daily, fig_status, fig_metrics

def create_download_package():
    """Create downloadable package with images and reports"""
    if not st.session_state.inspection_history:
        return None
    
    # Create ZIP file in memory
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add inspection report CSV
        df = pd.DataFrame(st.session_state.inspection_history)
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        zip_file.writestr('inspection_report.csv', csv_buffer.getvalue())
        
        # Add golden image if available
        if st.session_state.golden_image_data:
            golden_img_data = base64.b64decode(st.session_state.golden_image_data)
            zip_file.writestr('golden_reference.png', golden_img_data)
        
        # Add summary report
        summary = f"""
MSU Quality Inspector - Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Total Inspections: {len(df)}
Pass: {len(df[df['status'] == 'PASS'])} ({len(df[df['status'] == 'PASS'])/len(df)*100:.1f}%)
Warning: {len(df[df['status'] == 'WARNING'])} ({len(df[df['status'] == 'WARNING'])/len(df)*100:.1f}%)
Fail: {len(df[df['status'] == 'FAIL'])} ({len(df[df['status'] == 'FAIL'])/len(df)*100:.1f}%)

Average SSIM: {df['ssim'].mean():.3f}
Average MSE: {df['mse'].mean():.2f}
Average PSNR: {df['psnr'].mean():.2f}
"""
        zip_file.writestr('summary_report.txt', summary)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def login_page():
    """Display login/role selection page"""
    st.markdown('<div class="main-header">🔍 MSU Quality Inspector</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="login-container">
        <h3 style="text-align: center; margin-bottom: 2rem;">Select Your Role</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔧 Admin Portal", key="admin_btn", help="Upload golden images, manage settings, view analytics"):
            st.session_state.user_role = "admin"
            st.rerun()
        
        if st.button("👤 Quality Inspector", key="user_btn", help="Perform quality inspections"):
            st.session_state.user_role = "user"
            st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### 📋 Role Descriptions:
    
    **🔧 Admin Portal:**
    - Upload and manage golden reference images
    - Configure quality thresholds
    - View detailed analytics and reports
    - Download inspection data and images
    
    **👤 Quality Inspector:**
    - Quick photo capture and quality check
    - Instant pass/fail results
    - Mobile-optimized interface
    - View inspection history
    """)

def admin_portal():
    """Admin portal interface"""
    st.markdown('<div class="admin-header">🔧 Admin Portal</div>', unsafe_allow_html=True)
    
    # Logout button
    if st.button("← Back to Login", key="logout_admin"):
        st.session_state.user_role = None
        st.rerun()
    
    # Admin tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Golden Image", "⚙️ Settings", "📊 Analytics", "📥 Downloads"])
    
    with tab1:
        st.subheader("Golden Reference Management")
        
        # Display current golden image info
        if st.session_state.golden_image_info:
            st.success("✅ Golden image is configured")
            info = st.session_state.golden_image_info
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
                **Current Golden Image:**
                - Filename: {info['filename']}
                - Upload Date: {info['upload_date']}
                - Image Size: {info['image_size']}
                - Notes: {info.get('admin_notes', 'None')}
                """)
            
            with col2:
                golden_img = load_golden_image()
                if golden_img:
                    st.image(golden_img, caption="Current Golden Reference", use_column_width=True)
        else:
            st.warning("⚠️ No golden image configured. Please upload one below.")
        
        # Upload new golden image
        st.subheader("Upload New Golden Image")
        uploaded_file = st.file_uploader(
            "Choose golden MSU image",
            type=['jpg', 'jpeg', 'png'],
            help="This will be the reference for all quality comparisons"
        )
        
        admin_notes = st.text_area(
            "Admin Notes (Optional)",
            placeholder="Add any notes about this golden image..."
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Preview", use_column_width=True)
            
            if st.button("💾 Save as Golden Reference", type="primary"):
                save_golden_image(image, uploaded_file.name, admin_notes)
                st.success("✅ Golden image saved successfully!")
                st.rerun()
    
    with tab2:
        st.subheader("Quality Thresholds Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ssim_threshold = st.slider(
                "SSIM Threshold", 
                0.0, 1.0, 
                st.session_state.quality_thresholds['ssim'], 
                0.05,
                help="Structural Similarity threshold (higher = stricter)"
            )
            
            mse_threshold = st.slider(
                "MSE Threshold", 
                0, 5000, 
                st.session_state.quality_thresholds['mse'], 
                50,
                help="Mean Squared Error threshold (lower = stricter)"
            )
            
            max_diff = st.slider(
                "Max Differences", 
                0, 50, 
                st.session_state.quality_thresholds['max_differences'], 
                1,
                help="Maximum allowed difference regions"
            )
        
        with col2:
            st.info(f"""
            **Current Settings:**
            - SSIM Threshold: {ssim_threshold:.2f}
            - MSE Threshold: {mse_threshold}
            - Max Differences: {max_diff}
            
            **Recommendations:**
            - Stricter quality: Increase SSIM, decrease MSE
            - More lenient: Decrease SSIM, increase MSE
            """)
        
        if st.button("💾 Save Threshold Settings"):
            st.session_state.quality_thresholds = {
                'ssim': ssim_threshold,
                'mse': mse_threshold,
                'max_differences': max_diff
            }
            st.success("✅ Settings saved!")
    
    with tab3:
        st.subheader("Analytics Dashboard")
        
        if st.session_state.inspection_history:
            df = pd.DataFrame(st.session_state.inspection_history)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Inspections", len(df))
            
            with col2:
                pass_rate = len(df[df['status'] == 'PASS']) / len(df) * 100
                st.metric("Pass Rate", f"{pass_rate:.1f}%")
            
            with col3:
                avg_ssim = df['ssim'].mean()
                st.metric("Avg SSIM", f"{avg_ssim:.3f}")
            
            with col4:
                today_count = len(df[pd.to_datetime(df['timestamp']).dt.date == datetime.now().date()])
                st.metric("Today's Inspections", today_count)
            
            # Charts
            fig_daily, fig_status, fig_metrics = create_analytics_charts(df)
            
            if fig_daily:
                st.plotly_chart(fig_daily, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if fig_status:
                    st.plotly_chart(fig_status, use_container_width=True)
            
            with col2:
                if fig_metrics:
                    st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Recent inspections table
            st.subheader("Recent Inspections")
            recent_df = df.tail(10).sort_values('timestamp', ascending=False)
            st.dataframe(recent_df, use_container_width=True)
            
        else:
            st.info("📊 No inspection data available yet. Start using the quality inspector to see analytics.")
    
    with tab4:
        st.subheader("Download Reports & Data")
        
        if st.session_state.inspection_history:
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                df = pd.DataFrame(st.session_state.inspection_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV Report",
                    data=csv,
                    file_name=f"inspection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Complete package download
                zip_data = create_download_package()
                if zip_data:
                    st.download_button(
                        label="📦 Download Complete Package",
                        data=zip_data,
                        file_name=f"msu_inspection_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )
            
            # Clear data option
            st.subheader("⚠️ Data Management")
            if st.button("🗑️ Clear All Inspection Data", type="secondary"):
                if st.checkbox("I understand this will delete all inspection history"):
                    st.session_state.inspection_history = []
                    st.success("✅ All inspection data cleared!")
                    st.rerun()
        else:
            st.info("📥 No data available for download yet.")

def user_portal():
    """User portal interface"""
    st.markdown('<div class="user-header">👤 Quality Inspector Portal</div>', unsafe_allow_html=True)
    
    # Logout button
    if st.button("← Back to Login", key="logout_user"):
        st.session_state.user_role = None
        st.rerun()
    
    # Check if golden image is configured
    if not st.session_state.golden_image_data:
        st.error("❌ No golden reference image configured. Please contact admin to upload golden image first.")
        return
    
    st.success("✅ Golden reference loaded. Ready for inspections!")
    
    # Display golden image info
    with st.expander("📋 Golden Reference Info", expanded=False):
        info = st.session_state.golden_image_info
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Golden Reference:**
            - Filename: {info['filename']}
            - Upload Date: {info['upload_date']}
            - Notes: {info.get('admin_notes', 'None')}
            """)
        
        with col2:
            golden_img = load_golden_image()
            if golden_img:
                st.image(golden_img, caption="Golden Reference", use_column_width=True)
    
    # Test image upload
    st.subheader("📸 Upload MSU for Quality Check")
    
    st.markdown("""
    <div style="background: #f0f8ff; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
        <h4>📱 Mobile Users:</h4>
        <p>Use your phone's camera to take photos directly, or upload existing images from your gallery.</p>
    </div>
    """, unsafe_allow_html=True)
    
    test_file = st.file_uploader(
        "Upload Test MSU Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload the MSU image to be inspected - you can use your phone's camera!"
    )
    
    if test_file is not None:
        test_image = Image.open(test_file)
        golden_image = load_golden_image()
        
        # Display images
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🥇 Golden Reference")
            st.image(golden_image, caption="Reference Image", use_column_width=True)
        
        with col2:
            st.subheader("🔍 Test MSU")
            st.image(test_image, caption="Test Image", use_column_width=True)
        
        # Process images
        with st.spinner("🔄 Analyzing quality..."):
            golden_processed = preprocess_image(golden_image)
            test_processed = preprocess_image(test_image)
            
            metrics = calculate_similarity_metrics(golden_processed, test_processed)
            diff_overlay, difference_count = detect_differences(golden_processed, test_processed)
            
            status, status_icon = determine_quality_status(
                metrics, difference_count, st.session_state.quality_thresholds
            )
        
        # Display results
        st.subheader("📊 Quality Check Results")
        
        # Status display
        if status == "PASS":
            st.success(f"{status_icon} **{status}** - Quality Approved ✅")
        elif status == "WARNING":
            st.warning(f"{status_icon} **{status}** - Review Required ⚠️")
        else:
            st.error(f"{status_icon} **{status}** - Quality Rejected ❌")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("SSIM Score", f"{metrics['ssim']:.3f}")
        with col2:
            st.metric("Differences", difference_count)
        with col3:
            st.metric("MSE", f"{metrics['mse']:.0f}")
        with col4:
            st.metric("PSNR", f"{metrics['psnr']:.1f} dB")
        
        # Save inspection record
        if st.button("💾 Save Inspection Record", type="primary"):
            inspection_record = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': status,
                'ssim': metrics['ssim'],
                'mse': metrics['mse'],
                'psnr': metrics['psnr'],
                'ncc': metrics['ncc'],
                'differences': difference_count,
                'filename': test_file.name,
                'inspector': 'User'
            }
            
            st.session_state.inspection_history.append(inspection_record)
            st.success("✅ Inspection record saved!")
    
    # Quick stats for user
    if st.session_state.inspection_history:
        st.subheader("📈 Quick Stats")
        df = pd.DataFrame(st.session_state.inspection_history)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Inspections", len(df))
        
        with col2:
            pass_count = len(df[df['status'] == 'PASS'])
            pass_rate = pass_count / len(df) * 100
            st.metric("Pass Rate", f"{pass_rate:.1f}%")
        
        with col3:
            today_count = len(df[pd.to_datetime(df['timestamp']).dt.date == datetime.now().date()])
            st.metric("Today's Count", today_count)

# Main application
def main():
    initialize_session_state()
    
    if st.session_state.user_role is None:
        login_page()
    elif st.session_state.user_role == "admin":
        admin_portal()
    elif st.session_state.user_role == "user":
        user_portal()

if __name__ == "__main__":
    main()
