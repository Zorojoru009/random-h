# SDXL Lightning API Documentation

High-quality Baroque image generation API using SDXL Lightning 8-step UNet model.

## Features

- 🚀 **Ultra-fast generation**: 8-step inference vs traditional 50+ steps
- 🎨 **Maximum quality**: Uses UNet checkpoint (superior to LoRA)
- 📐 **16:9 aspect ratio**: Locked to 1024x576 for YouTube/video content
- 🔄 **Batch processing**: Generate up to 10 images per request
- 📚 **Auto-documentation**: Interactive Swagger UI at `/docs`

## Quick Start

### Running on Kaggle

1. **Create a new Kaggle Notebook** with GPU enabled (T4 recommended)

2. **Upload your files** to Kaggle:
   - `base.py`
   - `app.py`
   - `requirements.txt`

3. **Install dependencies**:
```bash
!pip install -q fastapi uvicorn[standard] pydantic diffusers transformers accelerate safetensors huggingface_hub
```

4. **Start the server**:
```python
# Run in background
!nohup python app.py > server.log 2>&1 &

# Wait for server to start
import time
time.sleep(10)

# Check if running
!curl http://localhost:8000/health
```

5. **Access the API** within your Kaggle notebook or via ngrok for external access

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
python app.py

# Or with uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

---

## API Endpoints

### `GET /`
**Root endpoint** - API information and available endpoints

**Response**:
```json
{
  "name": "SDXL Lightning API",
  "version": "1.0.0",
  "status": "running",
  "endpoints": {
    "generate": "/generate - POST - Generate baroque images from prompts",
    "health": "/health - GET - Health check",
    "docs": "/docs - GET - API documentation"
  }
}
```

---

### `GET /health`
**Health check** - Verify server status and GPU availability

**Response**:
```json
{
  "status": "healthy",
  "cuda_available": true,
  "cuda_device": "Tesla T4"
}
```

---

### `POST /generate`
**Generate images** - Create baroque-style images from text prompts

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompts` | `string[]` | ✅ Yes | List of text descriptions (1-10) |
| `negative_prompt` | `string` | ❌ No | What to avoid in images |
| `return_format` | `string` | ❌ No | `"base64"` or `"url"` (default: `"base64"`) |

#### Example Request

```json
{
  "prompts": [
    "baroque oil painting, dramatic chiaroscuro lighting, a massive disembodied eye made of fire, non-human, symbolic",
    "baroque oil painting, dramatic chiaroscuro lighting, a golden crown floating in a dark void, divine light"
  ],
  "negative_prompt": "modern, photograph, low quality, blurry",
  "return_format": "base64"
}
```

#### Example Response

```json
{
  "success": true,
  "images": [
    "iVBORw0KGgoAAAANSUhEUgAA...",
    "iVBORw0KGgoAAAANSUhEUgAA..."
  ],
  "count": 2,
  "message": "Successfully generated 2 image(s)"
}
```

---

## Usage Examples

### Python Client

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# API endpoint
API_URL = "http://localhost:8000/generate"

# Request payload
payload = {
    "prompts": [
        "baroque oil painting, dramatic chiaroscuro lighting, ancient philosopher contemplating mortality, dark background"
    ],
    "negative_prompt": "modern, photograph, cartoon, anime",
    "return_format": "base64"
}

# Make request
response = requests.post(API_URL, json=payload)
data = response.json()

# Decode and save images
for i, img_base64 in enumerate(data["images"]):
    img_data = base64.b64decode(img_base64)
    img = Image.open(BytesIO(img_data))
    img.save(f"output_{i}.png")
    print(f"Saved output_{i}.png")
```

### cURL

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["baroque oil painting, dramatic chiaroscuro lighting, a weathered stone tablet with ancient text"],
    "negative_prompt": "modern, low quality",
    "return_format": "base64"
  }'
```

### JavaScript/Fetch

```javascript
const generateImage = async () => {
  const response = await fetch('http://localhost:8000/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      prompts: [
        'baroque oil painting, dramatic chiaroscuro lighting, mystical symbols glowing in darkness'
      ],
      negative_prompt: 'modern, photograph',
      return_format: 'base64'
    })
  });
  
  const data = await response.json();
  
  // Convert base64 to image
  data.images.forEach((base64Img, i) => {
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${base64Img}`;
    document.body.appendChild(img);
  });
};

generateImage();
```

---

## Kaggle-Specific Setup

### Exposing API Externally with ngrok

If you need to access your API from outside Kaggle:

```python
# Install ngrok
!pip install pyngrok -q

# Start server in background
!nohup python app.py > server.log 2>&1 &

# Expose with ngrok
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")
```

### Setting Hugging Face Token (if needed)

```python
import os
os.environ["HF_TOKEN"] = "your_token_here"
```

### Memory Optimization

For Kaggle's 16GB VRAM limit, the current setup is optimized:
- Uses `torch.float16` for memory efficiency
- 8-step generation (fast + quality balance)
- Single model instance (cached globally)

---

## Error Handling

### Common HTTP Status Codes

| Code | Meaning | Solution |
|------|---------|----------|
| `200` | Success | Images generated successfully |
| `400` | Bad Request | Check prompt count (max 10) |
| `500` | Server Error | Check server logs, may be CUDA OOM |

### Example Error Response

```json
{
  "detail": "Image generation failed: CUDA out of memory"
}
```

**Solution**: Reduce batch size or restart Kaggle kernel to clear GPU memory.

---

## Performance Benchmarks

| GPU | Batch Size | Time per Image | Total Time (8 steps) |
|-----|------------|----------------|----------------------|
| T4 (Kaggle) | 1 | ~2-3s | ~2-3s |
| T4 (Kaggle) | 4 | ~2-3s | ~8-12s |

---

## Best Practices

### Prompt Engineering for Baroque Style

✅ **Good prompts**:
- "baroque oil painting, dramatic chiaroscuro lighting, [subject]"
- Include: "oil painting", "dramatic lighting", "chiaroscuro"
- Be specific about composition and mood

❌ **Avoid**:
- Modern references ("iPhone", "photograph")
- Generic descriptions
- Too many conflicting elements

### Negative Prompts

Recommended negative prompt:
```
"modern, photograph, 3d render, cartoon, anime, low quality, blurry, distorted"
```

---

## Troubleshooting

### Server won't start
```bash
# Check if port is in use
!lsof -i :8000

# Kill process if needed
!kill -9 <PID>
```

### CUDA out of memory
- Reduce batch size to 1-2 images
- Restart Kaggle kernel
- Check GPU usage: `!nvidia-smi`

### Model download slow
- Hugging Face models are cached after first download
- Subsequent runs will be much faster

---

## License & Credits

- **SDXL Lightning**: ByteDance ([Hugging Face](https://huggingface.co/ByteDance/SDXL-Lightning))
- **Stable Diffusion XL**: Stability AI
- **API Framework**: FastAPI

---

## Support

For issues or questions:
1. Check interactive docs: `http://localhost:8000/docs`
2. Review server logs: `!cat server.log` (Kaggle)
3. Verify GPU availability: `!nvidia-smi`
