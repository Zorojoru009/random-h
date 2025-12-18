"""
FastAPI server for SDXL Lightning Image Generation
Optimized for Kaggle environment with GPU support
"""

import os
# Fix for protobuf error often seen on Kaggle: 
# AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

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
# Import our image generation function
from base import generate_baroque_image, load_pipeline
from contextlib import asynccontextmanager

# Global variable for the model
ml_models = {}
import asyncio

class ModelManager:
    def __init__(self):
        self.models = []
        # Store (model, lock) tuples
        self.model_locks = []
        self.index = 0
        
    def add_model(self, pipe):
        self.models.append(pipe)
        self.model_locks.append(asyncio.Lock())
        
    def get_model_and_lock(self, index):
        """Get a specific model and its lock by index"""
        if index < 0 or index >= len(self.models):
            return None, None
        return self.models[index], self.model_locks[index]
        
    def get_next_model_and_lock(self):
        """Get the next model in rotation and its lock"""
        if not self.models:
            return None, None
        
        idx = self.index
        self.index = (self.index + 1) % len(self.models)
        
        return self.models[idx], self.model_locks[idx]

manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... (startup code remains similar, using add_model)
    # Load the ML models on startup
    print("🏗️ Pre-loading SDXL Lightning models on startup...")
    import torch
    
    try:
        from base import load_pipeline
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✨ Found {device_count} GPU(s)")
            
            for i in range(device_count):
                device = f"cuda:{i}"
                print(f"🚀 Loading model on {device}...")
                pipe = load_pipeline(device)
                manager.add_model(pipe)
                print(f"✅ Model on {device} loaded!")
        else:
            print("⚠️ No GPU detected, loading on CPU (slow!)")
            manager.add_model(load_pipeline("cpu"))
            
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print(traceback.format_exc())
        
    yield
    
    # Clean up
    manager.models.clear()
    manager.model_locks.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="SDXL Lightning API",
    description="High-quality Baroque image generation using SDXL Lightning 8-step UNet (Multi-GPU)",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    """Request model for image generation"""
    prompts: List[str] = Field(..., description="List of text prompts for image generation", min_items=1, max_items=150)
    negative_prompt: Optional[str] = Field("", description="Negative prompt to guide what to avoid")
    num_images_per_prompt: Optional[int] = Field(1, description="Number of images to generate per prompt (default: 1)", ge=1, le=50)
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
    """
    try:
        # Validate inputs
        total_images = len(request.prompts) * request.num_images_per_prompt
        if total_images > 150:
             raise HTTPException(status_code=400, detail=f"Total images ({total_images}) exceeds limit of 150 per request")
        
        # Expand prompts if num_images_per_prompt > 1
        expanded_prompts = []
        for prompt in request.prompts:
            expanded_prompts.extend([prompt] * request.num_images_per_prompt)
        
        # Split prompts for parallel processing if we have multiple models
        import torch
        from starlette.concurrency import run_in_threadpool
        num_models = len(manager.models)
        
        # If we have multiple models and enough prompts, split the work
        if num_models > 1 and len(expanded_prompts) >= num_models:
            print(f"⚡ Parallelizing {len(expanded_prompts)} images across {num_models} GPUs")
            
            # Split prompts into chunks
            chunk_size = len(expanded_prompts) // num_models
            chunks = []
            for i in range(num_models):
                start = i * chunk_size
                # Last chunk takes the remainder
                end = (i + 1) * chunk_size if i < num_models - 1 else len(expanded_prompts)
                chunks.append(expanded_prompts[start:end])
            
            # Define worker function with locking
            async def generate_chunk(chunk_prompts, model_index):
                pipe, lock = manager.get_model_and_lock(model_index)
                
                # Acquire lock for this specific GPU model
                async with lock:
                    return await run_in_threadpool(
                        generate_baroque_image,
                        prompts=chunk_prompts,
                        negative_prompt=request.negative_prompt,
                        save_to_disk=False,
                        pipe=pipe
                    )
            
            # Run tasks in parallel
            tasks = [generate_chunk(chunk, i) for i, chunk in enumerate(chunks) if chunk]
            results = await asyncio.gather(*tasks)
            
            # Flatten results
            images = [img for batch in results for img in batch]
            
        else:
            # Single model execution (or small batch)
            pipe, lock = manager.get_next_model_and_lock()
            
            if not pipe:
                # Fallback if no models loaded
                print("⚠️ Model not pre-loaded, performing just-in-time load...")
                pipe = load_pipeline()
                
                # No lock needed for a temporary independent pipeline
                images = await run_in_threadpool(
                    generate_baroque_image,
                    prompts=expanded_prompts,
                    negative_prompt=request.negative_prompt,
                    save_to_disk=False,
                    pipe=pipe
                )
            else:
                # Use the managed model with lock
                async with lock:
                    images = await run_in_threadpool(
                        generate_baroque_image,
                        prompts=expanded_prompts,
                        negative_prompt=request.negative_prompt,
                        save_to_disk=False,
                        pipe=pipe
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
    import subprocess
    import threading
    import time
    
    # Get port from environment (Kaggle uses different ports)
    port = int(os.getenv("PORT", 8000))
    use_public = os.getenv("USE_PUBLIC", "false").lower() == "true"
    
    print("="*70)
    print("🚀 SDXL Lightning API Server")
    print("="*70)
    
    # Setup zrok if enabled (for public access)
    zrok_process = None
    public_url = None
    
    if use_public:
        try:
            print("🔧 Setting up zrok tunnel...")
            
            # Start zrok share - it outputs the URL directly to stdout
            zrok_cmd = ["zrok", "share", "public", f"http://localhost:{port}", "--headless"]
            zrok_process = subprocess.Popen(
                zrok_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Read output to get the public URL
            print("⏳ Waiting for zrok tunnel to establish...")
            
            # Give zrok time to start and read its output
            import select
            timeout = 15  # seconds
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                # Check if process is still running
                if zrok_process.poll() is not None:
                    print("⚠️ zrok process exited unexpectedly")
                    break
                
                # Read available output
                line = zrok_process.stdout.readline()
                if line:
                    # Look for the URL in the output
                    if 'https://' in line and 'share.zrok.io' in line:
                        # Extract URL from line
                        import re
                        url_match = re.search(r'https://[^\s]+share\.zrok\.io', line)
                        if url_match:
                            public_url = url_match.group(0)
                            print(f"✅ zrok tunnel created!")
                            print(f"🌐 Public URL: {public_url}")
                            print(f"📚 Public API Docs: {public_url}/docs")
                            print(f"🔗 Share this URL to access your API from anywhere!")
                            break
                
                time.sleep(0.5)
            
            if not public_url:
                print(f"⚠️ zrok started but couldn't detect URL automatically")
                print(f"   The tunnel should still be running - check logs above")
            
            print("="*70)
            
        except FileNotFoundError:
            print("⚠️  zrok not found. Please install zrok:")
            print("    Visit: https://zrok.io/download")
            print("    Or run: curl -sSLf https://get.zrok.io | bash")
            print("    Then enable: zrok enable YOUR_TOKEN")
            print("    Running in local-only mode...")
            print("="*70)
            use_public = False
        except Exception as e:
            print(f"⚠️  Could not start zrok: {str(e)}")
            print("    Running in local-only mode...")
            print("="*70)
            use_public = False
    
    if not use_public:
        print(f"📍 Local URL: http://localhost:{port}")
        print(f"📚 API Documentation: http://localhost:{port}/docs")
        print(f"💡 Tip: Set USE_PUBLIC=true for public access via zrok")
        print("="*70)
    
    try:
        print("Starting server...")
        uvicorn.run(app, host="0.0.0.0", port=port)
    finally:
        # Cleanup zrok process on shutdown
        if zrok_process:
            print("\n🛑 Shutting down zrok tunnel...")
            zrok_process.terminate()
            zrok_process.wait(timeout=5)
