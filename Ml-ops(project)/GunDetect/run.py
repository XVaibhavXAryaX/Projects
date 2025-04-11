try:
    import uvicorn
    import fastapi
    import numpy
    import torch
    import PIL
    print("All required packages imported successfully!")
except ImportError as e:
    print(f"Error importing package: {str(e)}")
    print("Please make sure all packages are installed using:")
    print("pip install -r requirements.txt")
    exit(1)

if __name__ == "__main__":
    try:
        print("Starting FastAPI server...")
        uvicorn.run(
            "main:app",
            host="127.0.0.1",
            port=8000,
            reload=True
        )
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        print("Please make sure you're in the correct directory and all packages are installed.") 