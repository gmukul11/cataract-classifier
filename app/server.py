import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
from pathlib import Path
import io
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)
from components.train.src.model import get_model
import uvicorn


app = FastAPI(title="Cataract Classification API",
             description="API for detecting cataracts in eye images",
             version="1.0.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(model_name="mobilenet_v2",device=device)  # Load your best model
model.load_state_dict(torch.load('models/best_model_mobilenet_v2.pth', map_location=device)['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/api/v1/classify", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Make a cataract prediction on an uploaded image
    
    Parameters:
    - file: Image file upload
    
    Returns:
    - prediction: Cataract or Normal
    - confidence: Confidence score of the prediction
    """
    try:
        start = datetime.datetime.now()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        
        with torch.no_grad():
            outputs = model(image_tensor).squeeze()
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            predicted = torch.sigmoid(outputs) > 0.5
            confidence = round(float(probabilities),2)
            
        label = "cataract" if predicted.cpu().numpy() == 1 else "normal"
        
        return JSONResponse({ "success": True, 
            "response" : {
                "label": label,
                "confidence":  confidence},
            "diagnostics" : {"executionTime" : int((datetime.datetime.now() - start).microseconds / 1000)}
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success":False, "error": str(e)}
        )

@app.get("/health", tags=["Health"])
async def health_check():
    """Check if the API is running"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)