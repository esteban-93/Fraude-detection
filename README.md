# Fraud Detection API

A FastAPI-based fraud detection service for credit card transactions using machine learning models.

## 🏗️ Project Structure

```
Fraude-detection/
├── src/
│   └── fraud_detector/           # Main package
│       ├── models/               # ML models and preprocessing
│       │   ├── data_processor.py    # Data preprocessing
│       │   ├── training/            # Model training modules
│       │   ├── evaluation/          # Model evaluation modules
│       │   └── inference/           # Model inference modules
│       ├── api/                  # FastAPI endpoints
│       ├── core/                 # Core business logic
│       └── utils/                # Utilities and helpers
├── deployment/                   # Deployment configurations
│   ├── api/                      # API deployment files
│   └── config/                   # Deployment configs
├── configs/                      # Configuration files
├── scripts/                      # Utility scripts
├── models/                       # Trained model artifacts
├── data/                         # Data storage
├── logs/                         # Application logs
└── notebooks/                    # Jupyter notebooks for EDA
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/esteban-93/Fraude-detection.git
   cd Fraude-detection
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   export MODEL_PATH="models/trained/fraud_model.pkl"
   export SCALER_PATH="models/trained/scaler.pkl"
   ```

### Running the API

1. **Start the FastAPI server:**
   ```bash
   uvicorn deployment.api.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the API documentation:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## 📊 Dataset

**Source:** Real European credit card transactions from 2013, anonymized and transformed with PCA.

**Size:** 284,807 transactions, only 492 labeled as fraud.

### Features

- **Time:** Seconds since the first transaction
- **Amount:** Transaction amount
- **V1 to V28:** Transformed PCA components
- **Class:** Target variable (0 = genuine, 1 = fraud)

### Key Characteristics

- **Highly imbalanced dataset:** Only 0.172% of transactions are fraudulent
- **Anonymized features:** Original features were transformed using PCA for privacy
- **Time series data:** Transactions are ordered by time

## 🔧 API Endpoints

### Health Check
```http
GET /health
```

### Single Transaction Prediction
```http
POST /predict
Content-Type: application/json

{
  "Time": 0.0,
  "Amount": 149.62,
  "V1": -0.073403184690714,
  "V2": 0.27723157600516,
  // ... other V features
}
```

### Batch Prediction
```http
POST /predict_batch
Content-Type: application/json

{
  "transactions": [
    {
      "Time": 0.0,
      "Amount": 149.62,
      // ... transaction data
    }
  ]
}
```

### Model Information
```http
GET /model_info
```

## 🛠️ Development

### Training a New Model

```bash
python scripts/train_model.py
```

### Testing Predictions

```bash
python scripts/predict.py
```

### Configuration

Edit `configs/config.yaml` to modify:
- Model parameters
- API settings
- Data paths
- Logging configuration

## 📦 Dependencies

### Core ML Libraries
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- lightgbm >= 3.3.0
- xgboost >= 1.5.0

### API & Deployment
- fastapi >= 0.68.0
- uvicorn >= 0.15.0
- pydantic >= 1.8.0

### Data Processing
- imbalanced-learn >= 0.8.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## 🚀 Deployment

### Docker (Recommended)

```bash
# Build the image
docker build -t fraud-detection-api .

# Run the container
docker run -p 8000:8000 fraud-detection-api
```

### Production Deployment

1. **Using Gunicorn:**
   ```bash
   gunicorn deployment.api.fastapi_app:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **Using Docker Compose:**
   ```bash
   docker-compose up -d
   ```

## 📈 Model Performance

The fraud detection model achieves:
- **Precision:** 0.85+
- **Recall:** 0.80+
- **F1-Score:** 0.82+
- **ROC-AUC:** 0.95+

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions or support, please open an issue in the GitHub repository.

## �� Version History

- **v1.0.0** - Initial release with FastAPI service
- **v0.1.0** - Basic fraud detection models and preprocessing
