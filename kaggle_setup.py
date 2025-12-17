"""
Kaggle Notebook Setup Script for SDXL Lightning API
Run this in a Kaggle notebook with GPU enabled
"""

# ========================================
# STEP 1: Install Dependencies
# ========================================
print("📦 Installing dependencies...")
!pip install -q fastapi uvicorn[standard] pydantic diffusers transformers accelerate safetensors huggingface_hub

# ========================================
# STEP 2: Start API Server in Background
# ========================================
print("\n🚀 Starting SDXL Lightning API server...")
!nohup python app.py > server.log 2>&1 &

# Wait for server to initialize
import time
print("⏳ Waiting for server to start...")
time.sleep(15)

# ========================================
# STEP 3: Verify Server is Running
# ========================================
print("\n✅ Checking server health...")
!curl -s http://localhost:8000/health | python -m json.tool

# ========================================
# STEP 4: Test Image Generation
# ========================================
print("\n🎨 Testing image generation...")

import requests
import base64
from PIL import Image
from io import BytesIO
import json

# Test request
test_payload = {
    "prompts": [
        "baroque oil painting, dramatic chiaroscuro lighting, a massive disembodied eye made of fire, non-human, symbolic"
    ],
    "negative_prompt": "modern, photograph, low quality",
    "return_format": "base64"
}

try:
    response = requests.post("http://localhost:8000/generate", json=test_payload, timeout=120)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success! Generated {data['count']} image(s)")
        
        # Decode and display first image
        img_data = base64.b64decode(data["images"][0])
        img = Image.open(BytesIO(img_data))
        
        # Save and display
        img.save("test_output.png")
        print("💾 Saved as test_output.png")
        
        # Display in notebook
        from IPython.display import display
        display(img)
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Error: {str(e)}")
    print("\n📋 Check server logs:")
    !tail -50 server.log

# ========================================
# STEP 5: Show Access Information
# ========================================
print("\n" + "="*60)
print("🎉 SDXL Lightning API is ready!")
print("="*60)
print("📚 API Documentation: http://localhost:8000/docs")
print("🔍 Health Check: http://localhost:8000/health")
print("🎨 Generate Endpoint: POST http://localhost:8000/generate")
print("\n💡 For external access, use ngrok:")
print("   !pip install pyngrok")
print("   from pyngrok import ngrok")
print("   public_url = ngrok.connect(8000)")
print("="*60)

# ========================================
# Optional: Setup ngrok for external access
# ========================================
# Uncomment below to enable external access via ngrok

# !pip install -q pyngrok
# from pyngrok import ngrok
# 
# # Create public URL
# public_url = ngrok.connect(8000)
# print(f"\n🌐 Public URL: {public_url}")
# print(f"Access API docs at: {public_url}/docs")
