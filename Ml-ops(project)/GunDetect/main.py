import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
import logging
import sys
import socket

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

try:
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
model.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
])

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def predict_and_draw(image: Image.Image):
    try:
        img_tensor = transform(image).unsqueeze(0).to(device)
        logger.debug("Image transformed to tensor")

        with torch.no_grad():
            predictions = model(img_tensor)
            logger.debug("Predictions made")

        prediction = predictions[0]
        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        img_rgb = image.convert("RGB")
        draw = ImageDraw.Draw(img_rgb)

        for box, score in zip(boxes, scores):
            if score > 0.7:
                x_min, y_min, x_max, y_max = box
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        
        return img_rgb
    except Exception as e:
        logger.error(f"Error in predict_and_draw: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the Guns Object Detection API"}

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify the API is working"""
    return {
        "status": "API is working",
        "model_loaded": True,
        "device": str(device)
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
            
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Validate image
        if image.format not in ['JPEG', 'PNG']:
            raise HTTPException(status_code=400, detail="Image must be JPEG or PNG")
            
        output_image = predict_and_draw(image)

        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = 5000
    if is_port_in_use(port):
        logger.warning(f"Port {port} is in use, trying port {port + 1}")
        port += 1
    
    logger.info(f"Starting server on port {port}...")
    uvicorn.run(app, host="127.0.0.1", port=port)


