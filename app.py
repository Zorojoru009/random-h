"""
FastAPI server for SDXL Lightning Image Generation
Optimized for Kaggle environment with GPU support
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import base64
import io
import os
import traceback
from PIL import Image

# Import our image generation function
from base import generate_baroque_image

app = FastAPI(
    title="SDXL Lightning API",
    description="High-quality Baroque image generation using SDXL Lightning 8-step UNet",
    version="1.0.0"
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to cache the model (loaded once on first request)
_model_loaded = False

class GenerateRequest(BaseModel):
    """Request model for image generation"""
    prompts: List[str] = Field(..., description="List of text prompts for image generation", min_items=1, max_items=10)
    negative_prompt: Optional[str] = Field("", description="Negative prompt to guide what to avoid")
    return_format: Optional[str] = Field("base64", description="Response format: 'base64' or 'url'")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompts": [
                    "baroque oil painting, dramatic chiaroscuro lighting, a massive disembodied eye made of fire, non-human, symbolic"
                ],
                "negative_prompt": "modern, photograph, low quality, blurry",
                "return_format": "base64"
            }
        }

class GenerateResponse(BaseModel):
    """Response model for image generation"""
    success: bool
    images: List[str]
    count: int
    message: str

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "SDXL Lightning API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "generate": "/generate - POST - Generate baroque images from prompts",
            "health": "/health - GET - Health check",
            "docs": "/docs - GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    import torch
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_images(request: GenerateRequest):
    """
    Generate baroque-style images using SDXL Lightning.
    
    - **prompts**: List of text descriptions (1-10 prompts)
    - **negative_prompt**: What to avoid in the image
    - **return_format**: 'base64' returns base64-encoded PNG, 'url' returns file paths
    
    Returns a list of generated images in the specified format.
    """
    global _model_loaded
    
    try:
        # Validate inputs
        if len(request.prompts) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 prompts allowed per request")
        
        # Generate images (model will be loaded on first call)
        if not _model_loaded:
            print("Loading SDXL Lightning model (first time)... This may take a minute.")
            _model_loaded = True
        
        images = generate_baroque_image(
            prompts=request.prompts,
            negative_prompt=request.negative_prompt,
            save_to_disk=False  # Don't save to disk for API
        )
        
        # Convert images based on requested format
        result_images = []
        
        if request.return_format == "base64":
            # Convert PIL images to base64
            for img in images:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                result_images.append(img_str)
        else:
            # Save images and return file paths
            os.makedirs("outputs", exist_ok=True)
            for i, img in enumerate(images):
                filepath = f"outputs/generated_{i}.png"
                img.save(filepath)
                result_images.append(filepath)
        
        return GenerateResponse(
            success=True,
            images=result_images,
            count=len(result_images),
            message=f"Successfully generated {len(result_images)} image(s)"
        )
        
    except Exception as e:
        print(f"Error generating images: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Image generation failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (Kaggle uses different ports)
    port = int(os.getenv("PORT", 8000))
    
    print(f"Starting SDXL Lightning API server on port {port}...")
    print(f"API Documentation: http://localhost:{port}/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
