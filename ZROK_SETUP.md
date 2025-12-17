# Zrok Setup Guide for SDXL Lightning API

Zrok is an open-source tunneling solution that creates secure public URLs for your local services.

## Quick Setup (One-Time)

### 1. Install Zrok

**On Kaggle/Linux:**
```bash
!curl -sSLf https://get.zrok.io | bash
```

**On macOS:**
```bash
brew install zrok
```

**On Windows:**
Download from [https://zrok.io/download](https://zrok.io/download)

### 2. Get Your Free Account Token

1. Visit [https://zrok.io](https://zrok.io)
2. Click "Sign Up" (free account)
3. Copy your account token from the dashboard

### 3. Enable Zrok

Run this **once** to configure zrok with your token:

```bash
!zrok enable YOUR_TOKEN_HERE
```

Replace `YOUR_TOKEN_HERE` with the token you copied.

---

## Using Zrok with the API

### On Kaggle

```python
# 1. Setup zrok (one-time)
!curl -sSLf https://get.zrok.io | bash
!zrok enable YOUR_TOKEN_HERE

# 2. Enable public access
import os
os.environ["USE_PUBLIC"] = "true"

# 3. Run the API
!python app.py
```

### Locally

```bash
# 1. Install and enable zrok (one-time)
curl -sSLf https://get.zrok.io | bash
zrok enable YOUR_TOKEN_HERE

# 2. Run with public access
USE_PUBLIC=true python app.py
```

---

## What You'll See

When running with `USE_PUBLIC=true`:

```
======================================================================
🚀 SDXL Lightning API Server
======================================================================
🔧 Setting up zrok tunnel...
⏳ Waiting for zrok tunnel to establish...
✅ zrok tunnel created!
🌐 Public URL: https://abc123.share.zrok.io
📚 Public API Docs: https://abc123.share.zrok.io/docs
🔗 Share this URL to access your API from anywhere!
======================================================================
Starting server...
```

Share the public URL with anyone!

---

## Checking Zrok Status

```bash
# See all active zrok shares
zrok status

# Output shows:
# - Share ID
# - Public URL
# - Status
```

---

## Benefits of Zrok

✅ **100% Open Source** - No vendor lock-in  
✅ **No Rate Limits** - Unlimited tunnels  
✅ **Private by Default** - End-to-end encryption  
✅ **Free Forever** - No paid tiers needed  
✅ **Self-Hostable** - Run your own instance  

---

## Troubleshooting

### "zrok not found"
- Make sure installation completed successfully
- Restart your terminal/kernel
- Check: `which zrok`

### "zrok not enabled"
- Run: `zrok enable YOUR_TOKEN`
- Use the token from [zrok.io](https://zrok.io)

### Can't get public URL
- Wait 5-10 seconds after starting
- Run: `zrok status` to see active shares
- Check server logs for errors

---

## Additional Commands

```bash
# List all shares
zrok ls

# Disable a specific share
zrok disable \u003cshare-id\u003e

# Reset zrok config
zrok disable --reset
```

For more info: [https://docs.zrok.io](https://docs.zrok.io)
