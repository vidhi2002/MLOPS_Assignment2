import os
import sys
import logging
import time
import json
import io
from pathlib import Path
from typing import Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.cnn_model import ModelUtils, SimpleCNN

class ImprovedCNN(nn.Module):
    """Improved CNN model for Cats vs Dogs classification with better accuracy"""

    def __init__(self, num_classes=2):
        super(ImprovedCNN, self).__init__()

        # Enhanced architecture with more layers and better feature extraction
        self.features = nn.Sequential(
            # Block 1: 64 filters
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 2: 128 filters
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 3: 256 filters
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )

        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Enhanced classifier with more layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x
from src.data.preprocessing import DataPreprocessor

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics - clear registry first to avoid duplicates
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, REGISTRY, CollectorRegistry

# Create a fresh registry
registry = CollectorRegistry()
REQUEST_COUNT = Counter('inference_requests_total', 'Total inference requests', ['method', 'endpoint'], registry=registry)
REQUEST_LATENCY = Histogram('inference_request_duration_seconds', 'Inference request latency', registry=registry)
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made', ['predicted_class'], registry=registry)

# Initialize FastAPI app
app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="ML model API for binary image classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
preprocessor = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_mapping = {0: 'cat', 1: 'dog'}

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

def load_model():
    """Load the trained model"""
    global model, preprocessor, device
    
    try:
        model_path = os.getenv('MODEL_PATH', 'models/best_model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint to determine model type
        checkpoint = torch.load(model_path, map_location='cpu')
        model_class_name = checkpoint.get('model_class', 'SimpleCNN')
        
        if model_class_name == 'ImprovedCNN':
            model = ImprovedCNN(num_classes=2)
        else:
            model = SimpleCNN(num_classes=2)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        preprocessor = DataPreprocessor(data_dir="data/processed")
        logger.info("Model loaded successfully", 
                   model_type=model_class_name,
                   device=str(device))
        return True
        
    except Exception as e:
        logger.error("Failed to load model", error=str(e))
        return False

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for inference"""
    try:
        # Resize to 224x224
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return tensor.to(device)
        
    except Exception as e:
        logger.error("Image preprocessing failed", error=str(e))
        raise HTTPException(status_code=400, detail="Invalid image format")

@REQUEST_LATENCY.time()
def predict_image(image_tensor: torch.Tensor) -> Dict[str, Any]:
    """Make prediction on preprocessed image"""
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            # Convert to numpy for easier handling
            probabilities = probabilities.cpu().numpy()[0]
            predicted_class = predicted_class.item()
            confidence = confidence.item()
            
            # Create response
            class_probs = {
                'cat': float(probabilities[0]),
                'dog': float(probabilities[1])
            }
            
            result = {
                'predicted_class': class_mapping[predicted_class],
                'confidence': confidence,
                'probabilities': class_probs
            }
            
            # Update metrics
            PREDICTION_COUNT.labels(predicted_class=class_mapping[predicted_class]).inc()
            
            logger.info("Prediction made", 
                       predicted_class=result['predicted_class'],
                       confidence=confidence)
            
            return result
            
    except Exception as e:
        logger.error("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting Cats vs Dogs inference service")
    success = load_model()
    if not success:
        logger.warning("Model not loaded, but service will start for health checks")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    REQUEST_COUNT.labels(method="GET", endpoint="/health").inc()
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device) if device else "unknown"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predict endpoint for image upload"""
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    
    start_time = time.time()
    
    try:
        # Validate file type - check if file exists and has content type
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Debug: print file info
        print(f"File info: filename={file.filename}, content_type={file.content_type}, size={file.size}")
        
        # Check content type or file extension
        content_type = file.content_type
        if not content_type:
            # Fallback to checking file extension
            if file.filename and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                content_type = 'image/jpeg' if file.filename.lower().endswith(('.jpg', '.jpeg')) else 'image/png'
        
        if not content_type or not content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"File must be an image, got content_type: {content_type}")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess and predict
        image_tensor = preprocess_image(image)
        result = predict_image(image_tensor)
        
        processing_time = time.time() - start_time
        logger.info("Request processed", 
                   filename=file.filename,
                   processing_time=processing_time)
        
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Request processing failed", 
                    filename=file.filename,
                    error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    REQUEST_COUNT.labels(method="GET", endpoint="/metrics").inc()
    try:
        # Use our custom registry
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        # If metrics fail, return a simple text response
        return Response("# HELP inference_requests_total Total inference requests\n# TYPE inference_requests_total counter\ninference_requests_total{method=\"GET\",endpoint=\"/metrics\"} 1", media_type="text/plain")

@app.get("/")
async def root():
    """Root endpoint"""
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    return {
        "message": "Cats vs Dogs Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics"
        }
    }

if __name__ == "__main__":
    import io
    
    # Load model before starting server
    if not load_model():
        print("Failed to load model. Exiting...")
        exit(1)
    
    # Run server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        log_level="info"
    )
