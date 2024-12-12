# Cataract Classification Project

This project implements a deep learning solution for cataract detection in eye images, including model training and API deployment.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Kaggle API credentials (for dataset download)

## Installation

1. Clone the repository:
```bash
git clone git@github.com:gmukul11/cataract-classifier.git
cd cataract-classifier
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate 
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Kaggle credentials:
- Download kaggle.json from your Kaggle account
- Place it in ~/.kaggle/ directory
```bash
mkdir -p ~/.kaggle
cp path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```
## Steps to Run

1. Download Data
```bash
#!/bin/bash
curl -L -o ./data/cataract-image-dataset.zip https://www.kaggle.com/api/v1/datasets/download/nandanp6/cataract-image-dataset
```
2. UnZip the Data inside data folder
3. Run model training component
```bash
python3 components/train/src/component.py
```

## Indiviudal ML Pipeline

1. Data Source:
```bash
python3 components/source/src/component.py
```
This will:
- Create dataloader
- Split data into train and test sets

2. Data Transformation:
```bash
python3 components/transform/src/component.py
```
This performs:
- Image preprocessing
- Data augmentation
- Creates transformed dataset

3. Model Training:
```bash
python3 components/train/src/component.py
```
Features:
- Trains multiple models (VanillaCNN, ResNet50, VGG16, etc.)
- Saves model checkpoints
- Logs metrics using MLflow

4. Model Evaluation:
```bash
python components/validation/src/component.py
```
Generates:
- Performance metrics
- Confusion matrices
- ROC curves

## API Deployment

1. Start the API server:
```bash
# Option 1: Using Python
python3 app/server.py

# Option 2: Using Uvicorn directly
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

2. API Endpoints:
- Health Check: `GET /health`
- Prediction: `POST /api/v1/classify`

3. Example API Usage:
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Make prediction
with open("path/to/image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/classify",
        files={"file": f}
    )
print(response.json())
```

## Running Tests

Run the test suite:
```bash
# Run specific tests
pytest tests/test_api.py
```

## Model Training Results

The project includes several models:
- VanillaCNN
- ResNet50
- VGG16
- MobileNetV2

Best performing model metrics:
- Accuracy: 98.3%
- Precision: 0.98
- Recall: 0.98
- F1-Score: 0.98
  
- ![confusion_matrix_mobilenet_v2](https://github.com/user-attachments/assets/c9c879c1-552a-4c7c-9fa2-4b5f3780be46)




## MLflow Tracking

View training metrics and experiments:
1. Start MLflow server:
```bash
mlflow ui
```
2. Access dashboard at: http://localhost:5000

## Configuration

Modify `config.yml` to adjust:
- Model architectures
- Training parameters


## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit changes
4. Push to the branch
5. Create Pull Request
