# 🔍 MSU Quality Inspector

A Streamlit web application for quality control in MSU (Manufacturing Support Unit) assembly lines. This app allows quality inspectors to compare assembled MSUs against a golden reference using computer vision and machine learning.

## 🌟 Features

- **Image Comparison**: Compare test MSUs against golden reference
- **Mobile-Friendly**: Optimized for mobile phone usage on assembly lines
- **Quality Metrics**: SSIM, MSE, PSNR, and NCC analysis
- **Visual Difference Detection**: Automatic highlighting of defects
- **Inspection History**: Track and export quality control records
- **Real-time Analysis**: Instant pass/fail determination

## 🚀 Quick Start

### Online Demo
Visit: [Your Streamlit App URL will appear here after deployment]

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/msu-quality-inspector.git
   cd msu-quality-inspector
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run msu_quality_inspector.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 📱 Mobile Usage

1. Open the app URL in your mobile browser
2. Upload golden MSU reference image (one-time setup)
3. Use camera to take photos of MSUs for inspection
4. Get instant quality results with visual feedback
5. Export inspection reports as needed

## 🔧 Configuration

### Quality Thresholds
- **SSIM Threshold**: Structural similarity (default: 0.85)
- **MSE Threshold**: Mean squared error (default: 1000)
- **Difference Count**: Maximum allowed differences (default: 5)

### Supported Image Formats
- JPG/JPEG
- PNG
- Direct camera capture on mobile devices

## 📊 Quality Metrics Explained

- **SSIM (Structural Similarity)**: Measures structural similarity (0-1, higher is better)
- **MSE (Mean Squared Error)**: Pixel-wise differences (lower is better)
- **PSNR (Peak Signal-to-Noise Ratio)**: Signal quality measure (higher is better)
- **NCC (Normalized Cross-Correlation)**: Template matching score (0-1, higher is better)

## 🏭 Assembly Line Integration

This app is designed for:
- Quality control stations
- Mobile inspection workflows
- Real-time defect detection
- Quality data collection and reporting

## 🛠️ Technical Requirements

- Python 3.7+
- Streamlit
- OpenCV
- scikit-image
- Plotly
- Pandas

## 📈 Quality Control Workflow

1. **Setup**: Upload golden MSU reference image
2. **Inspection**: Capture/upload test MSU images
3. **Analysis**: Automatic computer vision analysis
4. **Decision**: Pass/Warning/Fail determination
5. **Documentation**: Save inspection records
6. **Reporting**: Export quality data for analysis

## 🔒 Security & Privacy

- No data stored permanently on servers
- Images processed in-session only
- Inspection history can be exported locally
- Mobile-secure HTTPS connections

## 📞 Support

For technical support or feature requests:
- Create an issue in this repository
- Contact: [Your contact information]

## 📝 License

[Add your preferred license]

---

**Built for manufacturing excellence** 🏭✨