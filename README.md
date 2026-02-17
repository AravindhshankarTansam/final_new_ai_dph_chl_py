# 💧 Chlorine Detection System

A machine learning-based system to detect chlorine concentration (ppm) in water samples from images.

## 🎯 Features

- **Universal Container Detection**: Works with tubes, bottles, cups, and various containers
- **Blur Detection**: Automatic image quality assessment
- **Preprocessing Pipeline**: White balance, lighting normalization, skin removal
- **Liquid Extraction**: Smart detection of liquid region in images
- **PPM Prediction**: Predicts chlorine concentration in 9 ranges (0-4+ ppm)
- **Streamlit Web App**: User-friendly interface for predictions

## 📊 Model Details

- **Algorithm**: LightGBM Regressor
- **Features**: 7 features extracted from LAB/HSV color space
- **Training**: Class-weighted sampling to handle imbalanced data
- **Output**: Continuous PPM value mapped to ranges

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/chlorine-detection.git
cd chlorine-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
chlorine-detection/
│
├── feature_extractor.py    # Image preprocessing & feature extraction
├── ppm_model.py            # Model loading & prediction
├── app.py                  # Streamlit web application
├── pipeline_v7.py          # Batch processing script
├── metrics.py              # Model evaluation script
├── ppm_model.pkl           # Trained LightGBM model
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🎮 Usage

### Web Application

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Batch Processing

```bash
python pipeline_v7.py
```

Edit the `INPUT_FOLDER` path in the script to process your images.

### Model Evaluation

```bash
python metrics.py
```

## 📊 PPM Ranges

| Range | Label | Safety |
|-------|-------|--------|
| 0 - 0.5 | Safe | Under-chlorinated |
| 0.5 - 1 | Normal | ✅ |
| 1 - 1.5 | Normal | ✅ |
| 1.5 - 2 | Normal | ✅ |
| 2 - 2.5 | Moderate | ✅ |
| 2.5 - 3 | Moderate | ⚠️ |
| 3 - 3.5 | High | ⚠️ |
| 3.5 - 4 | High | ⚠️ |
| 4+ | Over-chlorinated | ❌ |

## 🧪 Image Quality Requirements

- **Blur Score**: > 35 for best results (Laplacian variance)
- **Lighting**: Works in varied lighting (uses CLAHE normalization)
- **Container**: Any transparent/translucent container
- **Liquid Color**: Yellow, orange, or reddish tones

## 🛠️ Technical Details

### Feature Extraction Pipeline

1. **White Balance**: Normalize color temperature
2. **Lighting Normalization**: CLAHE on LAB color space
3. **Skin Removal**: Filter out hand/skin regions (HSV thresholding)
4. **Liquid Detection**: Color-based segmentation (yellow/red ranges)
5. **Feature Computation**: Extract 7 color and spatial features

### Features Used

1. LAB L* channel mean (lightness)
2. LAB a* channel mean (green-red)
3. LAB b* channel mean (blue-yellow)
4. HSV Hue mean
5. HSV Saturation mean
6. Standard deviation (color uniformity)
7. Area ratio (liquid region size)

## 📝 License

MIT License

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

## 📧 Contact

For questions or issues, please open a GitHub issue.
