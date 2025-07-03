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
    page_title="MSU Quality Control System | Robotic Warehouse Automation",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="auto"
)

# Professional CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background with overlay */
    .main > div {
        background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                    url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 800"><rect fill="%23f0f2f6" width="1200" height="800"/><g fill="%23ff6b35" opacity="0.1"><circle cx="200" cy="200" r="30"/><circle cx="400" cy="300" r="25"/><circle cx="800" cy="150" r="35"/><rect x="600" y="400" width="60" height="40" rx="5"/><rect x="300" y="500" width="80" height="50" rx="8"/></g><text x="100" y="100" font-family="Arial" font-size="24" fill="%23666" opacity="0.3">ROBOTIC WAREHOUSE AUTOMATION</text></svg>') center/cover;
        background-attachment: fixed;
        min-height: 100vh;
    }
    
    /* Header styles */
    .enterprise-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .enterprise-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .enterprise-header .subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    .admin-header {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(220, 38, 38, 0.3);
        margin-bottom: 2rem;
    }
    
    .user-header {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(5, 150, 105, 0.3);
        margin-bottom: 2rem;
    }
    
    /* Card styles */
    .professional-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%);
        backdrop-filter: blur(15px);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(226, 232, 240, 0.5);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .status-pass {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .status-fail {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    /* Login portal */
    .login-portal {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        max-width: 500px;
        margin: 3rem auto;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .role-selection {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .role-card {
        flex: 1;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .admin-role {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-color: #fecaca;
    }
    
    .admin-role:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(220, 38, 38, 0.3);
    }
    
    .user-role {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-color: #bbf7d0;
    }
    
    .user-role:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white;
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(5, 150, 105, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.9);
        border: 2px dashed #d1d5db;
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #3b82f6;
        background: rgba(59, 130, 246, 0.05);
    }
    
    /* Quality indicators */
    .quality-indicator {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .enterprise-header h1 {
            font-size: 2rem;
        }
        
        .enterprise-header .subtitle {
            font-size: 1rem;
        }
        
        .professional-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .login-portal {
            margin: 1rem;
            padding: 2rem;
        }
        
        .role-selection {
            flex-direction: column;
        }
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #64748b;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
    }
    
    /* Analytics styling */
    .analytics-summary {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(226, 232, 240, 0.5);
    }
    
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0.5rem 0;
    }
    
    .kpi-label {
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.9rem;
    }
    
    /* Company branding */
    .company-footer {
        background: rgba(30, 41, 59, 0.95);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 3rem;
        backdrop-filter: blur(15px);
    }
    
    .company-footer .brand {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ff6b35;
        margin-bottom: 0.5rem;
    }
    
    .company-footer .tagline {
        color: #94a3b8;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# File-based storage functions
def get_data_file_path(filename):
    """Get path for data files"""
    return f"./{filename}"

def save_to_file(data, filename):
    """Save data to JSON file"""
    try:
        with open(get_data_file_path(filename), 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return False

def load_from_file(filename):
    """Load data from JSON file"""
    try:
        with open(get_data_file_path(filename), 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Initialize session state with file-based persistence
def initialize_session_state():
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    
    # Load golden image data from file
    if 'golden_image_data' not in st.session_state:
        golden_data = load_from_file('golden_image.json')
        st.session_state.golden_image_data = golden_data.get('image_data') if golden_data else None
    
    if 'golden_image_info' not in st.session_state:
        golden_data = load_from_file('golden_image.json')
        st.session_state.golden_image_info = golden_data.get('info', {}) if golden_data else {}
    
    # Load inspection history from file
    if 'inspection_history' not in st.session_state:
        history_data = load_from_file('inspection_history.json')
        st.session_state.inspection_history = history_data if history_data else []
    
    # Load quality thresholds from file
    if 'quality_thresholds' not in st.session_state:
        threshold_data = load_from_file('quality_thresholds.json')
        st.session_state.quality_thresholds = threshold_data if threshold_data else {
            'ssim': 0.85,
            'mse': 1000,
            'max_differences': 5
        }

def save_golden_image(image, filename, admin_notes=""):
    """Save golden image to persistent file storage"""
    # Convert PIL image to base64 for storage
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Create data structure
    golden_data = {
        'image_data': img_str,
        'info': {
            'filename': filename,
            'upload_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'admin_notes': admin_notes,
            'image_size': image.size
        }
    }
    
    # Save to file and update session state
    if save_to_file(golden_data, 'golden_image.json'):
        st.session_state.golden_image_data = img_str
        st.session_state.golden_image_info = golden_data['info']
        return True
    return False

def save_inspection_record(record):
    """Save inspection record to persistent storage"""
    st.session_state.inspection_history.append(record)
    save_to_file(st.session_state.inspection_history, 'inspection_history.json')

def save_quality_thresholds(thresholds):
    """Save quality thresholds to persistent storage"""
    st.session_state.quality_thresholds = thresholds
    save_to_file(thresholds, 'quality_thresholds.json')

def clear_all_data():
    """Clear all persistent data"""
    import os
    files_to_remove = ['golden_image.json', 'inspection_history.json', 'quality_thresholds.json']
    
    for filename in files_to_remove:
        try:
            os.remove(get_data_file_path(filename))
        except FileNotFoundError:
            pass  # File doesn't exist, which is fine
    
    # Reset session state
    st.session_state.golden_image_data = None
    st.session_state.golden_image_info = {}
    st.session_state.inspection_history = []
    st.session_state.quality_thresholds = {
        'ssim': 0.85,
        'mse': 1000,
        'max_differences': 5
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
    """Create professional analytics charts"""
    if df.empty:
        return None, None, None
    
    # Daily inspection count with professional styling
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_counts = df.groupby('date').size().reset_index(name='count')
    
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(
        x=daily_counts['date'], 
        y=daily_counts['count'],
        mode='lines+markers',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=8, color='#1d4ed8'),
        name='Daily Inspections',
        hovertemplate='<b>Date:</b> %{x}<br><b>Inspections:</b> %{y}<extra></extra>'
    ))
    
    fig_daily.update_layout(
        title='Daily Quality Inspections Trend',
        xaxis_title='Date',
        yaxis_title='Number of Inspections',
        template='plotly_white',
        showlegend=False,
        font=dict(family="Inter", size=12),
        title_font=dict(size=16, color='#1e293b'),
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Pass/Fail distribution with custom colors
    status_counts = df['status'].value_counts()
    colors = {'PASS': '#10b981', 'WARNING': '#f59e0b', 'FAIL': '#ef4444'}
    
    fig_status = go.Figure(data=[go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        marker_colors=[colors.get(status, '#64748b') for status in status_counts.index],
        textinfo='label+percent',
        textfont=dict(size=12, family="Inter"),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig_status.update_layout(
        title='Quality Status Distribution',
        template='plotly_white',
        font=dict(family="Inter", size=12),
        title_font=dict(size=16, color='#1e293b'),
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Quality metrics trend
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['ssim'],
        mode='lines+markers',
        line=dict(color='#059669', width=2),
        marker=dict(size=6),
        name='SSIM Score',
        hovertemplate='<b>Time:</b> %{x}<br><b>SSIM:</b> %{y:.3f}<extra></extra>'
    ))
    
    fig_metrics.update_layout(
        title='Structural Similarity Quality Trend',
        xaxis_title='Timestamp',
        yaxis_title='SSIM Score',
        template='plotly_white',
        showlegend=False,
        font=dict(family="Inter", size=12),
        title_font=dict(size=16, color='#1e293b'),
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
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
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        zip_file.writestr('inspection_report.csv', csv_buffer.getvalue())
        
        # Add golden image if available
        if st.session_state.golden_image_data:
            golden_img_data = base64.b64decode(st.session_state.golden_image_data)
            zip_file.writestr('golden_reference.png', golden_img_data)
        
        # Add executive summary report
        summary = f"""
MSU QUALITY CONTROL SYSTEM - EXECUTIVE REPORT
Robotic Warehouse Automation Division
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

============================================
EXECUTIVE SUMMARY
============================================

Total Quality Inspections: {len(df)}
Pass Rate: {len(df[df['status'] == 'PASS'])/len(df)*100:.1f}%
Warning Rate: {len(df[df['status'] == 'WARNING'])/len(df)*100:.1f}%
Failure Rate: {len(df[df['status'] == 'FAIL'])/len(df)*100:.1f}%

============================================
QUALITY METRICS
============================================

Average Structural Similarity (SSIM): {df['ssim'].mean():.4f}
Average Mean Squared Error (MSE): {df['mse'].mean():.2f}
Average Peak Signal-to-Noise Ratio (PSNR): {df['psnr'].mean():.2f}

============================================
OPERATIONAL INSIGHTS
============================================

- Quality control system operational since: {df['timestamp'].min()}
- Latest inspection: {df['timestamp'].max()}
- Most recent SSIM score: {df['ssim'].iloc[-1]:.4f}
- System reliability: {'Excellent' if len(df[df['status'] == 'PASS'])/len(df) > 0.9 else 'Good' if len(df[df['status'] == 'PASS'])/len(df) > 0.8 else 'Needs Attention'}

============================================
RECOMMENDATIONS
============================================

{"• Maintain current quality standards - excellent performance" if len(df[df['status'] == 'PASS'])/len(df) > 0.9 else "• Review quality thresholds and processes" if len(df[df['status'] == 'PASS'])/len(df) < 0.8 else "• Continue monitoring - stable performance"}
• Regular calibration of golden reference images recommended
• Consider automated reporting for continuous improvement

Report generated by MSU Quality Control System v2.0
Robotic Warehouse Automation Division
"""
        zip_file.writestr('executive_summary.txt', summary)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def login_page():
    """Professional login/role selection page"""
    st.markdown("""
    <div class="enterprise-header">
        <h1>🏭 MSU Quality Control System</h1>
        <div class="subtitle">Robotic Warehouse Automation | Enterprise Quality Management</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="login-portal">
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #1e293b; margin-bottom: 0.5rem;">Access Control Portal</h2>
            <p style="color: #64748b;">Select your authorization level to continue</p>
        </div>
        
        <div class="role-selection">
            <div class="role-card admin-role" onclick="streamlit.setComponentValue('admin')">
                <div style="font-size: 2rem; margin-bottom: 1rem;">🔧</div>
                <h3 style="margin: 0.5rem 0; color: #dc2626;">System Administrator</h3>
                <p style="margin: 0; font-size: 0.9rem; color: #64748b;">Full system access & configuration</p>
            </div>
            
            <div class="role-card user-role" onclick="streamlit.setComponentValue('user')">
                <div style="font-size: 2rem; margin-bottom: 1rem;">👨‍🔬</div>
                <h3 style="margin: 0.5rem 0; color: #059669;">Quality Inspector</h3>
                <p style="margin: 0; font-size: 0.9rem; color: #64748b;">Quality control & inspection</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        admin_btn = st.button("🔧 System Administrator", key="admin_btn", help="Access: Golden image management, analytics, system configuration", use_container_width=True)
        user_btn = st.button("👨‍🔬 Quality Inspector", key="user_btn", help="Access: Quality inspections, real-time analysis", use_container_width=True)
        
        if admin_btn:
            st.session_state.user_role = "admin"
            st.rerun()
        
        if user_btn:
            st.session_state.user_role = "user"
            st.rerun()
    
    # Professional footer
    st.markdown("""
    <div class="company-footer">
        <div class="brand">ROBOTIC WAREHOUSE AUTOMATION</div>
        <div class="tagline">Precision • Efficiency • Innovation</div>
        <div style="margin-top: 1rem; font-size: 0.8rem; color: #64748b;">
            MSU Quality Control System v2.0 | Enterprise Edition
        </div>
    </div>
    """, unsafe_allow_html=True)

def admin_portal():
    """Professional admin portal interface"""
    st.markdown("""
    <div class="admin-header">
        <h1>🔧 System Administrator Portal</h1>
        <div class="subtitle">Enterprise Quality Control Management</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional logout button
    col1, col2, col3 = st.columns([1, 4, 1])
    with col3:
        if st.button("🚪 Logout", key="logout_admin"):
            st.session_state.user_role = None
            st.rerun()
    
    # Professional admin tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Reference Management", "⚙️ System Configuration", "📊 Analytics Dashboard", "📋 Reports & Export"])
    
    with tab1:
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.subheader("🎯 Golden Reference Image Management")
        
        # Current status indicator
        if st.session_state.golden_image_info:
            st.markdown("""
            <div class="status-badge status-pass">
                ✅ REFERENCE IMAGE CONFIGURED
            </div>
            """, unsafe_allow_html=True)
            
            info = st.session_state.golden_image_info
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📋 Current Configuration</h4>
                    <div class="quality-indicator">
                        <span><strong>Filename:</strong></span>
                        <span>{info['filename']}</span>
                    </div>
                    <div class="quality-indicator">
                        <span><strong>Upload Date:</strong></span>
                        <span>{info['upload_date']}</span>
                    </div>
                    <div class="quality-indicator">
                        <span><strong>Resolution:</strong></span>
                        <span>{info['image_size'][0]} x {info['image_size'][1]} px</span>
                    </div>
                    <div class="quality-indicator">
                        <span><strong>Admin Notes:</strong></span>
                        <span>{info.get('admin_notes', 'None specified')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                golden_img = load_golden_image()
                if golden_img:
                    st.image(golden_img, caption="🥇 Active Golden Reference", use_column_width=True)
        else:
            st.markdown("""
            <div class="status-badge status-fail">
                ❌ NO REFERENCE IMAGE CONFIGURED
            </div>
            """, unsafe_allow_html=True)
            st.warning("⚠️ System requires golden reference image configuration before quality inspections can begin.")
        
        st.markdown("---")
        
        # Upload new reference
        st.subheader("📤 Upload New Golden Reference")
        
        uploaded_file = st.file_uploader(
            "Select Golden Reference Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload the master reference image for quality comparisons"
        )
        
        admin_notes = st.text_area(
            "Technical Notes & Documentation",
            placeholder="Document any specific details about this reference image, version, or configuration notes...",
            height=100
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption="📋 Preview - New Reference", use_column_width=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📊 Image Analysis</h4>
                    <div class="quality-indicator">
                        <span><strong>Resolution:</strong></span>
                        <span>{image.size[0]} x {image.size[1]} pixels</span>
                    </div>
                    <div class="quality-indicator">
                        <span><strong>Format:</strong></span>
                        <span>{image.format}</span>
                    </div>
                    <div class="quality-indicator">
                        <span><strong>Mode:</strong></span>
                        <span>{image.mode}</span>
                    </div>
                    <div class="quality-indicator">
                        <span><strong>File Size:</strong></span>
                        <span>{len(uploaded_file.getvalue()) / 1024:.1f} KB</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("💾 Deploy as Golden Reference", type="primary", use_container_width=True):
                if save_golden_image(image, uploaded_file.name, admin_notes):
                    st.success("✅ Golden reference deployed successfully across all inspection stations!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Deployment failed. Please contact system administrator.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Quality Control Parameters")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**🎯 Tolerance Settings**")
            
            ssim_threshold = st.slider(
                "Structural Similarity Index (SSIM)", 
                0.0, 1.0, 
                st.session_state.quality_thresholds['ssim'], 
                0.01,
                help="Higher values = stricter quality control"
            )
            
            mse_threshold = st.slider(
                "Mean Squared Error (MSE)", 
                0, 5000, 
                st.session_state.quality_thresholds['mse'], 
                25,
                help="Lower values = stricter quality control"
            )
            
            max_diff = st.slider(
                "Maximum Defect Regions", 
                0, 50, 
                st.session_state.quality_thresholds['max_differences'], 
                1,
                help="Maximum number of detected difference regions"
            )
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Current Configuration</h4>
                <div class="quality-indicator">
                    <span><strong>SSIM Threshold:</strong></span>
                    <span>{ssim_threshold:.3f}</span>
                </div>
                <div class="quality-indicator">
                    <span><strong>MSE Threshold:</strong></span>
                    <span>{mse_threshold}</span>
                </div>
                <div class="quality-indicator">
                    <span><strong>Max Defects:</strong></span>
                    <span>{max_diff}</span>
                </div>
                <br>
                <h4>💡 Quality Level Assessment</h4>
                <div class="status-badge {'status-pass' if ssim_threshold >= 0.9 else 'status-warning' if ssim_threshold >= 0.8 else 'status-fail'}">
                    {'🔒 STRICT' if ssim_threshold >= 0.9 else '⚖️ BALANCED' if ssim_threshold >= 0.8 else '🔓 LENIENT'} Quality Control
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("💾 Apply Configuration Changes", type="primary", use_container_width=True):
            save_quality_thresholds({
                'ssim': ssim_threshold,
                'mse': mse_threshold,
                'max_differences': max_diff
            })
            st.success("✅ Configuration updated and synchronized across all inspection stations!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.subheader("📊 Enterprise Analytics Dashboard")
        
        if st.session_state.inspection_history:
            df = pd.DataFrame(st.session_state.inspection_history)
            
            # Executive KPI Dashboard
            st.markdown("### 📈 Key Performance Indicators")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value">{len(df)}</div>
                    <div class="kpi-label">Total Inspections</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                pass_rate = len(df[df['status'] == 'PASS']) / len(df) * 100
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value" style="color: {'#10b981' if pass_rate >= 90 else '#f59e0b' if pass_rate >= 80 else '#ef4444'}">{pass_rate:.1f}%</div>
                    <div class="kpi-label">Pass Rate</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_ssim = df['ssim'].mean()
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value">{avg_ssim:.3f}</div>
                    <div class="kpi-label">Avg Quality Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                today_count = len(df[pd.to_datetime(df['timestamp']).dt.date == datetime.now().date()])
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value">{today_count}</div>
                    <div class="kpi-label">Today's Inspections</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Professional Charts
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
            
            # Recent Activity Table
            st.markdown("### 📋 Recent Inspection Activity")
            recent_df = df.tail(15).sort_values('timestamp', ascending=False)
            
            # Style the dataframe
            def style_status(val):
                if val == 'PASS':
                    return 'background-color: #dcfce7; color: #166534; font-weight: bold'
                elif val == 'WARNING':
                    return 'background-color: #fef3c7; color: #92400e; font-weight: bold'
                else:
                    return 'background-color: #fee2e2; color: #991b1b; font-weight: bold'
            
            styled_df = recent_df.style.applymap(style_status, subset=['status'])
            st.dataframe(styled_df, use_container_width=True, height=400)
            
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #64748b;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
                <h3>No Inspection Data Available</h3>
                <p>Analytics will appear once quality inspections begin.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.subheader("📋 Enterprise Reporting & Data Export")
        
        if st.session_state.inspection_history:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Standard Reports**")
                
                df = pd.DataFrame(st.session_state.inspection_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📈 Download CSV Report",
                    data=csv,
                    file_name=f"MSU_Quality_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.markdown("**📦 Complete Data Package**")
                zip_data = create_download_package()
                if zip_data:
                    st.download_button(
                        label="📁 Download Executive Package",
                        data=zip_data,
                        file_name=f"MSU_Executive_Package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📋 Report Contents</h4>
                    <div class="quality-indicator">
                        <span>• Detailed inspection records</span>
                    </div>
                    <div class="quality-indicator">
                        <span>• Golden reference image</span>
                    </div>
                    <div class="quality-indicator">
                        <span>• Executive summary report</span>
                    </div>
                    <div class="quality-indicator">
                        <span>• Quality metrics analysis</span>
                    </div>
                    <div class="quality-indicator">
                        <span>• Operational recommendations</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Advanced Data Management
            st.markdown("**⚠️ Advanced Data Management**")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.warning("⚠️ **CAUTION:** This action will permanently delete all system data including golden reference images, inspection history, and configuration settings.")
            
            with col2:
                if st.button("🗑️ Purge All Data", type="secondary"):
                    confirm = st.checkbox("✅ I confirm data purge authorization", key="confirm_purge")
                    if confirm:
                        clear_all_data()
                        st.success("✅ System data purged successfully!")
                        st.rerun()
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #64748b;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📋</div>
                <h3>No Data Available for Export</h3>
                <p>Reports will be available once inspection data is collected.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def user_portal():
    """Professional user portal interface"""
    st.markdown("""
    <div class="user-header">
        <h1>👨‍🔬 Quality Inspector Portal</h1>
        <div class="subtitle">Real-Time MSU Quality Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional logout
    col1, col2, col3 = st.columns([1, 4, 1])
    with col3:
        if st.button("🚪 Logout", key="logout_user"):
            st.session_state.user_role = None
            st.rerun()
    
    # System status check
    if not st.session_state.golden_image_data:
        st.markdown("""
        <div class="professional-card">
            <div class="status-badge status-fail">
                ❌ SYSTEM NOT READY
            </div>
            <h3>🔧 Configuration Required</h3>
            <p>The quality control system requires golden reference image configuration by a system administrator before inspections can begin.</p>
            <p><strong>Action Required:</strong> Contact your system administrator to configure the golden reference image.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # System ready indicator
    st.markdown("""
    <div class="status-badge status-pass">
        ✅ SYSTEM OPERATIONAL
    </div>
    """, unsafe_allow_html=True)
    
    # Reference info in expandable section
    with st.expander("📋 View Active Golden Reference", expanded=False):
        info = st.session_state.golden_image_info
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Reference Configuration</h4>
                <div class="quality-indicator">
                    <span><strong>Reference ID:</strong></span>
                    <span>{info['filename']}</span>
                </div>
                <div class="quality-indicator">
                    <span><strong>Deployed:</strong></span>
                    <span>{info['upload_date']}</span>
                </div>
                <div class="quality-indicator">
                    <span><strong>Resolution:</strong></span>
                    <span>{info['image_size'][0]} x {info['image_size'][1]} px</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            golden_img = load_golden_image()
            if golden_img:
                st.image(golden_img, caption="🥇 Active Golden Reference", use_column_width=True)
    
    # Main inspection interface
    st.markdown('<div class="professional-card">', unsafe_allow_html=True)
    st.subheader("📸 MSU Quality Inspection")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                padding: 1.5rem; border-radius: 12px; margin: 1rem 0; border-left: 4px solid #0ea5e9;">
        <h4 style="margin: 0 0 0.5rem 0; color: #0c4a6e;">📱 Mobile Inspection Instructions</h4>
        <p style="margin: 0; color: #075985;">
            • Position MSU in good lighting for optimal image capture<br>
            • Ensure complete MSU visibility in frame<br>
            • Use device camera for real-time capture or upload existing images
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    test_file = st.file_uploader(
        "Upload MSU Image for Quality Analysis",
        type=['jpg', 'jpeg', 'png'],
        help="Capture or upload MSU image for automated quality inspection"
    )
    
    if test_file is not None:
        test_image = Image.open(test_file)
        golden_image = load_golden_image()
        
        # Image comparison display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🥇 Golden Reference**")
            st.image(golden_image, caption="Master Reference Standard", use_column_width=True)
        
        with col2:
            st.markdown("**🔍 Test Specimen**")
            st.image(test_image, caption="Current MSU Under Inspection", use_column_width=True)
        
        # Analysis processing
        with st.spinner("🔄 Performing Quality Analysis..."):
            golden_processed = preprocess_image(golden_image)
            test_processed = preprocess_image(test_image)
            
            metrics = calculate_similarity_metrics(golden_processed, test_processed)
            diff_overlay, difference_count = detect_differences(golden_processed, test_processed)
            
            status, status_icon = determine_quality_status(
                metrics, difference_count, st.session_state.quality_thresholds
            )
        
        # Results display
        st.markdown("---")
        st.subheader("📊 Quality Analysis Results")
        
        # Main status indicator
        if status == "PASS":
            st.markdown("""
            <div class="status-badge status-pass">
                ✅ QUALITY APPROVED - PASS
            </div>
            """, unsafe_allow_html=True)
        elif status == "WARNING":
            st.markdown("""
            <div class="status-badge status-warning">
                ⚠️ QUALITY WARNING - REVIEW REQUIRED
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-badge status-fail">
                ❌ QUALITY REJECTED - FAIL
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Similarity Score", 
                f"{metrics['ssim']:.3f}",
                delta=f"{metrics['ssim'] - st.session_state.quality_thresholds['ssim']:.3f}"
            )
        with col2:
            st.metric(
                "Detected Defects", 
                difference_count,
                delta=f"{difference_count - st.session_state.quality_thresholds['max_differences']}"
            )
        with col3:
            st.metric("MSE Score", f"{metrics['mse']:.0f}")
        with col4:
            st.metric("PSNR", f"{metrics['psnr']:.1f} dB")
        
        # Save record
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("💾 Save Inspection", type="primary", use_container_width=True):
                inspection_record = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'status': status,
                    'ssim': metrics['ssim'],
                    'mse': metrics['mse'],
                    'psnr': metrics['psnr'],
                    'ncc': metrics['ncc'],
                    'differences': difference_count,
                    'filename': test_file.name,
                    'inspector': 'Quality Inspector'
                }
                
                save_inspection_record(inspection_record)
                st.success("✅ Inspection record saved to enterprise database!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Inspector dashboard
    if st.session_state.inspection_history:
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.subheader("📈 Inspector Performance Dashboard")
        
        df = pd.DataFrame(st.session_state.inspection_history)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{len(df)}</div>
                <div class="kpi-label">Total Inspections</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            pass_count = len(df[df['status'] == 'PASS'])
            pass_rate = pass_count / len(df) * 100
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color: {'#10b981' if pass_rate >= 90 else '#f59e0b' if pass_rate >= 80 else '#ef4444'}">{pass_rate:.1f}%</div>
                <div class="kpi-label">Pass Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            today_count = len(df[pd.to_datetime(df['timestamp']).dt.date == datetime.now().date()])
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{today_count}</div>
                <div class="kpi-label">Today's Count</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

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
