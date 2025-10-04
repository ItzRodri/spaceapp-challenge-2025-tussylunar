# Exoplanet AI Classifier

**NASA Space Apps Challenge 2025** - Advanced AI/ML model for automatic exoplanet classification using Kepler and TESS data.

## 🌟 Features

### 🎯 Dual Mode Interface
- **Basic Mode**: Quick classification for individual exoplanet candidates
- **Expert Mode**: Batch processing for researchers with advanced analytics

### 🤖 Advanced AI/ML
- **XGBoost** classifier with calibration
- **SHAP** explanations for interpretability
- **Multi-mission** training (Kepler KOI + TESS TOI)
- **Calibrated probabilities** with confidence levels

### 📊 Comprehensive Analytics
- ROC-AUC and PR-AUC metrics
- Reliability diagrams and calibration analysis
- Feature importance rankings
- Mission-specific performance metrics

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```

### 3. Run the Application
```bash
# Option 1: Use the start script (recommended)
./start_server.sh

# Option 2: Manual start
source venv/bin/activate
python main.py
```

### 4. Access the Web Interface
Open your browser and navigate to `http://localhost:8001`

## 📁 Project Structure

```
spaceAppChallenge2025/
├── src/
│   ├── models/
│   │   └── exoplanet_classifier.py    # Main ML model
│   ├── data/
│   │   └── preprocessor.py            # Data preprocessing
│   └── utils/
│       ├── validators.py              # Input validation
│       └── metrics.py                 # Metrics calculation
├── templates/
│   ├── base.html                      # Base template
│   ├── index.html                     # Home page
│   ├── basic_mode.html               # Basic mode interface
│   └── expert_mode.html              # Expert mode interface
├── static/
│   └── style.css                     # Custom styles
├── models/                           # Trained models (created after training)
├── main.py                           # FastAPI application
├── train_model.py                    # Model training script
└── requirements.txt                  # Dependencies
```

## 🎮 Usage

### Basic Mode
1. Navigate to `/basic`
2. Enter exoplanet parameters:
   - Orbital period (days)
   - Transit duration (hours)
   - Transit depth (ppm)
   - Stellar radius (solar)
   - Stellar temperature (K)
   - Signal-to-noise ratio (optional)
3. Select mission (Kepler/TESS)
4. Get instant prediction with SHAP explanations

### Expert Mode
1. Navigate to `/expert`
2. Download CSV template
3. Upload CSV file with multiple candidates
4. View batch results with filtering options
5. Export results and view detailed metrics

## 📊 Model Performance

The model achieves:
- **High accuracy** on exoplanet classification
- **Calibrated probabilities** for reliable confidence estimates
- **Interpretable predictions** with SHAP explanations
- **Cross-mission generalization** (Kepler → TESS)

## 🔬 Technical Details

### Data Sources
- **Kepler Object of Interest (KOI)** dataset
- **TESS Object of Interest (TOI)** dataset
- Combined training for robust generalization

### Feature Engineering
- Log transformations for skewed features
- Mission-specific normalization
- Ratio features for enhanced discrimination
- Outlier clipping and imputation

### Model Architecture
- **XGBoost** gradient boosting classifier
- **Isotonic calibration** for probability calibration
- **SHAP TreeExplainer** for feature importance
- **Stratified cross-validation** by mission and class

### Classification Classes
- **CONFIRMED**: Validated exoplanets
- **CANDIDATE**: Potential exoplanets requiring follow-up
- **FALSE_POSITIVE**: Non-planetary signals

## 🛠️ API Endpoints

- `GET /` - Home page
- `GET /basic` - Basic mode interface
- `GET /expert` - Expert mode interface
- `POST /api/predict/single` - Single prediction
- `POST /api/predict/batch` - Batch prediction
- `GET /api/metrics` - Model performance metrics
- `GET /api/template` - Download CSV template

## 📈 Future Enhancements

- [ ] Active learning with user feedback
- [ ] Real-time model updates
- [ ] Additional mission support (K2, CHEOPS)
- [ ] Advanced visualization dashboards
- [ ] Model ensemble methods
- [ ] Uncertainty quantification

## 🤝 Contributing

This project was developed for the NASA Space Apps Challenge 2025. For questions or contributions, please contact the development team.

## 📄 License

This project is part of the NASA Space Apps Challenge 2025 and follows the challenge guidelines and requirements.

## 🙏 Acknowledgments

- NASA Exoplanet Archive for providing the datasets
- Kepler and TESS mission teams
- NASA Space Apps Challenge organizers
- Open source ML community (XGBoost, SHAP, FastAPI)

---

**Built with ❤️ for NASA Space Apps Challenge 2025**
