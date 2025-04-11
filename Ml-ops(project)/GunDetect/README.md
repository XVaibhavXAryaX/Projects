# Gun Detection API

A FastAPI-based application that uses a pre-trained Faster R-CNN model to detect guns in images. The API processes uploaded images and returns them with detected guns marked with red rectangles.

## Features

- Real-time gun detection using pre-trained Faster R-CNN model
- RESTful API endpoints for image processing
- Support for JPEG and PNG image formats
- Automatic device selection (CPU/GPU)
- Confidence threshold-based detection

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/GunDetect.git
cd GunDetect
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the API server:
```bash
python run.py
```

2. The API will be available at:
- Main endpoint: http://127.0.0.1:8000
- Test endpoint: http://127.0.0.1:8000/test
- API documentation: http://127.0.0.1:8000/docs

3. To detect guns in an image:
   - Use the `/predict` endpoint
   - Send a POST request with an image file
   - The API will return the processed image with detected guns marked

### Example using Postman:

1. Create a new POST request to: http://127.0.0.1:8000/predict
2. In the request body:
   - Select "form-data"
   - Add a key named "file"
   - Set the type to "File"
   - Select an image file
3. Send the request
4. The API will return the processed image with detected guns marked in red rectangles

## API Endpoints

- `GET /`: Welcome message
- `GET /test`: Test endpoint to verify API status
- `POST /predict`: Process an image and detect guns

## Technical Details

- Model: Faster R-CNN with ResNet-50 backbone
- Confidence threshold: 0.7
- Supported image formats: JPEG, PNG
- Device: Automatically uses GPU if available, falls back to CPU

## Requirements

- Python 3.8+
- FastAPI
- PyTorch
- torchvision
- Pillow
- numpy
- uvicorn

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Uses pre-trained models from torchvision
- Built with FastAPI framework
- Inspired by the need for automated gun detection systems 