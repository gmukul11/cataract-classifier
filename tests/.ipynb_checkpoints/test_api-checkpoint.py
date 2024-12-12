import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
import io
from PIL import Image
import numpy as np

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from app.server import app

client = TestClient(app)

@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a simple RGB test image
    img = Image.new('RGB', (224, 224), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint_success(sample_image):
    """Test successful prediction with valid image"""
    files = {
        "file": ("test_image.png", sample_image, "image/png")
    }
    response = client.post("/api/v1/classify", files=files)
    assert response.status_code == 200
    
    # Check response structure
    json_response = response.json()
    assert json_response["success"] == True
    assert "response" in json_response
    assert "label" in json_response["response"]
    assert "confidence" in json_response["response"]
    assert "diagnostics" in json_response
    assert "executionTime" in json_response["diagnostics"]
    
    # Verify data types
    assert isinstance(json_response["response"]["label"], str)
    assert isinstance(json_response["response"]["confidence"], float)
    assert isinstance(json_response["diagnostics"]["executionTime"], int)
    
    # Check label values
    assert json_response["response"]["label"] in ["normal", "cataract"]
    assert 0 <= json_response["response"]["confidence"] <= 1

def test_predict_endpoint_no_file():
    """Test prediction endpoint with no file"""
    response = client.post("/api/v1/classify")
    assert response.status_code == 422  # FastAPI's default response for missing required fields

def test_predict_endpoint_large_image(tmp_path):
    """Test prediction with a large image"""
    # Create a large RGB image
    large_img = Image.new('RGB', (2000, 2000), color='white')
    img_path = tmp_path / "large_image.png"
    large_img.save(img_path)
    
    with open(img_path, "rb") as img_file:
        files = {
            "file": ("large_image.png", img_file, "image/png")
        }
        response = client.post("/api/v1/classify", files=files)
        assert response.status_code == 200
        assert response.json()["success"] == True

def test_model_input_normalization(sample_image):
    """Test if the model handles various image formats"""
    files = {
        "file": ("test_image.png", sample_image, "image/png")
    }
    response = client.post("/api/v1/classify", files=files)
    assert response.status_code == 200

    confidence = response.json()["response"]["confidence"]
    assert 0 <= confidence <= 1

if __name__ == "__main__":
    pytest.main(["-v"])