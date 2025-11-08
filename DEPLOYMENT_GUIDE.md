# 🚀 Deployment Guide - Tesla Financial Model

## 📋 **Overview**

This guide covers deploying your Tesla Financial Model application to free hosting platforms. The app consists of:
- **Backend:** FastAPI (Python) on port 8002
- **Frontend:** React application
- **Database:** MongoDB
- **Vector DB:** ChromaDB (local)
- **External APIs:** OpenAI API

---

## 🆓 **Free Hosting Platforms Comparison**

| Platform | Backend | Frontend | Database | Free Tier | Best For |
|----------|---------|----------|----------|-----------|----------|
| **Railway** | ✅ | ✅ | ✅ (MongoDB Atlas) | $5 credit/month | **Recommended** - Easiest full-stack |
| **Render** | ✅ | ✅ | ❌ (Use MongoDB Atlas) | Free tier available | Good alternative |
| **Fly.io** | ✅ | ✅ | ❌ (Use MongoDB Atlas) | 3 VMs free | Good for scaling |
| **Vercel** | ⚠️ (Serverless) | ✅ | ❌ | Generous free tier | Frontend + API routes |
| **Netlify** | ⚠️ (Functions) | ✅ | ❌ | Generous free tier | Frontend + Functions |
| **PythonAnywhere** | ✅ | ❌ | ❌ | Free tier limited | Backend only |
| **MongoDB Atlas** | N/A | N/A | ✅ | 512MB free | Database only |

---

## 🏆 **Recommended: Railway (Easiest Full-Stack)**

### **Why Railway?**
- ✅ Deploys both backend and frontend
- ✅ Automatic HTTPS
- ✅ Environment variables management
- ✅ $5 free credit/month (usually enough for small apps)
- ✅ Easy MongoDB Atlas integration
- ✅ Auto-deploy from Git

### **Step 1: Prepare Your Code**

1. **Ensure your code is on GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Create `Procfile` for backend** (if not exists):
   ```bash
   # Create backend/Procfile
   web: uvicorn server:app --host 0.0.0.0 --port $PORT
   ```

3. **Create `runtime.txt` for backend:**
   ```bash
   # Create backend/runtime.txt
   python-3.11
   ```

### **Step 2: Set Up MongoDB Atlas (Free)**

1. Go to https://www.mongodb.com/cloud/atlas
2. Sign up for free account
3. Create a free cluster (M0 - 512MB)
4. Create database user
5. Whitelist IP: `0.0.0.0/0` (allow all IPs for Railway)
6. Get connection string (e.g., `mongodb+srv://user:pass@cluster.mongodb.net/dbname`)

### **Step 3: Deploy Backend to Railway**

1. Go to https://railway.app
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway will auto-detect Python
6. Configure:
   - **Root Directory:** `backend`
   - **Start Command:** `uvicorn server:app --host 0.0.0.0 --port $PORT`
   - **Python Version:** 3.11

7. **Add Environment Variables:**
   ```
   MONGO_URL=mongodb+srv://user:pass@cluster.mongodb.net/dbname
   DB_NAME=tesla_financial_model
   OPENAI_API_KEY=your_openai_api_key
   BACKEND_PORT=8002
   FRONTEND_URL=https://your-frontend.railway.app
   PORT=${{PORT}}  # Railway sets this automatically
   ```

8. **Deploy** - Railway will automatically:
   - Install dependencies from `requirements.txt`
   - Start your FastAPI server
   - Provide a public URL (e.g., `https://your-backend.railway.app`)

### **Step 4: Deploy Frontend to Railway**

1. In Railway, click "New Service" → "GitHub Repo"
2. Select same repository
3. Configure:
   - **Root Directory:** `frontend`
   - **Build Command:** `npm install && npm run build`
   - **Start Command:** `npx serve -s build -l $PORT`

4. **Add Environment Variables:**
   ```
   REACT_APP_BACKEND_URL=https://your-backend.railway.app
   PORT=${{PORT}}
   ```

5. **Deploy** - Railway will build and serve your React app

### **Step 5: Update CORS Settings**

Update `backend/server.py` CORS settings:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-frontend.railway.app",  # Add your Railway frontend URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🎯 **Alternative: Render (Free Tier)**

### **Deploy Backend to Render**

1. Go to https://render.com
2. Sign up with GitHub
3. Click "New" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Name:** `tesla-backend`
   - **Environment:** `Python 3`
   - **Build Command:** `cd backend && pip install -r requirements.txt`
   - **Start Command:** `cd backend && uvicorn server:app --host 0.0.0.0 --port $PORT`

6. **Environment Variables:**
   ```
   MONGO_URL=mongodb+srv://...
   DB_NAME=tesla_financial_model
   OPENAI_API_KEY=your_key
   PORT=${{PORT}}
   ```

7. **Free Tier Limitations:**
   - Spins down after 15 minutes of inactivity
   - Takes ~30 seconds to wake up
   - 750 hours/month free

### **Deploy Frontend to Render**

1. Click "New" → "Static Site"
2. Connect repository
3. Configure:
   - **Build Command:** `cd frontend && npm install && npm run build`
   - **Publish Directory:** `frontend/build`

4. **Environment Variables:**
   ```
   REACT_APP_BACKEND_URL=https://your-backend.onrender.com
   ```

---

## 🌐 **Alternative: Vercel (Frontend) + Railway (Backend)**

### **Deploy Frontend to Vercel**

1. Go to https://vercel.com
2. Sign up with GitHub
3. Import your repository
4. Configure:
   - **Framework Preset:** Create React App
   - **Root Directory:** `frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `build`

5. **Environment Variables:**
   ```
   REACT_APP_BACKEND_URL=https://your-backend.railway.app
   ```

6. **Deploy** - Vercel provides instant deployment and CDN

**Note:** Vercel's backend functions are serverless and may not work well with your FastAPI app. Use Railway for backend.

---

## 🔧 **Required Configuration Files**

### **1. Backend: `Procfile` (for Railway/Render)**
```bash
# backend/Procfile
web: uvicorn server:app --host 0.0.0.0 --port $PORT
```

### **2. Backend: `runtime.txt` (optional)**
```bash
# backend/runtime.txt
python-3.11
```

### **3. Frontend: `vercel.json` (for Vercel)**
```json
{
  "version": 2,
  "builds": [
    {
      "src": "frontend/package.json",
      "use": "@vercel/static-build",
      "config": {
        "distDir": "build"
      }
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/frontend/$1"
    }
  ]
}
```

### **4. Frontend: `netlify.toml` (for Netlify)**
```toml
[build]
  command = "cd frontend && npm install && npm run build"
  publish = "frontend/build"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

---

## 🔐 **Environment Variables Setup**

### **Backend Environment Variables**

Create these in your hosting platform:

```bash
# MongoDB
MONGO_URL=mongodb+srv://username:password@cluster.mongodb.net/dbname
DB_NAME=tesla_financial_model

# OpenAI
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini  # Optional

# Server
BACKEND_PORT=8002
PORT=${{PORT}}  # Platform sets this

# CORS
FRONTEND_URL=https://your-frontend-url.com

# Optional
PYTHONUNBUFFERED=1
```

### **Frontend Environment Variables**

```bash
REACT_APP_BACKEND_URL=https://your-backend-url.com
```

**Important:** In React, environment variables must start with `REACT_APP_` to be accessible in the browser.

---

## 📝 **Pre-Deployment Checklist**

### **Code Changes Needed:**

1. **Update CORS in `backend/server.py`:**
   ```python
   allow_origins=[
       "http://localhost:3000",
       os.getenv("FRONTEND_URL", ""),  # Add production URL
   ]
   ```

2. **Update API URL in frontend:**
   - Already using `process.env.REACT_APP_BACKEND_URL` ✅
   - Just set environment variable in hosting platform

3. **Fix hardcoded paths:**
   - Check `backend/services/analytics_engine.py` for hardcoded paths
   - Use relative paths or environment variables

4. **Update data file paths:**
   - Ensure CSV/Excel files are in `backend/data/`
   - Use relative paths: `Path(__file__).parent.parent / "data" / "file.csv"`

### **Files to Create:**

- [ ] `backend/Procfile`
- [ ] `backend/runtime.txt` (optional)
- [ ] `frontend/vercel.json` (if using Vercel)
- [ ] `frontend/netlify.toml` (if using Netlify)

### **Environment Variables:**

- [ ] MongoDB Atlas connection string
- [ ] OpenAI API key
- [ ] Frontend URL
- [ ] Backend URL

---

## 🗄️ **Database Setup (MongoDB Atlas)**

### **Free Tier Includes:**
- 512MB storage
- Shared RAM
- Free forever (no credit card required for M0)

### **Setup Steps:**

1. **Create Cluster:**
   - Choose M0 (Free) tier
   - Select region closest to your hosting
   - Create cluster (takes 3-5 minutes)

2. **Create Database User:**
   - Database Access → Add New User
   - Username/password
   - Set permissions: "Read and write to any database"

3. **Network Access:**
   - Network Access → Add IP Address
   - Add `0.0.0.0/0` (allow all - for hosting platforms)
   - Or add specific Railway/Render IPs

4. **Get Connection String:**
   - Clusters → Connect → Connect your application
   - Copy connection string
   - Replace `<password>` with your password
   - Replace `<dbname>` with your database name

---

## 🚨 **Important Notes**

### **Free Tier Limitations:**

1. **Railway:**
   - $5 credit/month (usually enough for small apps)
   - May need to upgrade for production

2. **Render:**
   - Free tier spins down after 15 min inactivity
   - First request after spin-down takes ~30 seconds
   - 750 hours/month free

3. **Vercel:**
   - 100GB bandwidth/month
   - Unlimited requests
   - Serverless functions: 100GB-hours/month

4. **MongoDB Atlas:**
   - 512MB storage
   - Shared resources
   - May be slow under load

### **ChromaDB (Vector Database):**

ChromaDB runs locally and stores data in `backend/data/chroma_db/`. For production:
- **Option 1:** Use ChromaDB Cloud (paid)
- **Option 2:** Keep local storage (data persists in Railway/Render volumes)
- **Option 3:** Use alternative: Pinecone (free tier available)

### **File Uploads:**

User uploads are stored in `backend/data/uploads/`. For production:
- Use cloud storage (AWS S3, Cloudinary - free tiers available)
- Or use Railway/Render persistent volumes

---

## 🔄 **Auto-Deployment Setup**

### **Railway Auto-Deploy:**

1. Connect GitHub repository
2. Railway auto-deploys on every push to `main` branch
3. Configure in Railway dashboard → Settings → Source

### **Render Auto-Deploy:**

1. Connect GitHub repository
2. Auto-deploy enabled by default
3. Configure branch in Render dashboard

### **Vercel Auto-Deploy:**

1. Connect GitHub repository
2. Auto-deploys on every push
3. Preview deployments for PRs

---

## 🧪 **Testing Deployment**

### **1. Test Backend:**

```bash
# Check if backend is running
curl https://your-backend.railway.app/api/health

# Or visit in browser
https://your-backend.railway.app/docs  # FastAPI docs
```

### **2. Test Frontend:**

```bash
# Visit frontend URL
https://your-frontend.railway.app

# Check browser console for errors
# Verify API calls are going to correct backend URL
```

### **3. Test Database Connection:**

- Check backend logs in Railway/Render dashboard
- Look for MongoDB connection messages
- Test an endpoint that uses database

---

## 🐛 **Troubleshooting**

### **Backend Won't Start:**

1. **Check logs:**
   - Railway: Deployments → View logs
   - Render: Logs tab

2. **Common issues:**
   - Missing environment variables
   - Port not set correctly (use `$PORT`)
   - Dependencies not installing
   - Python version mismatch

### **Frontend Can't Connect to Backend:**

1. **Check CORS:**
   - Ensure frontend URL is in CORS allow_origins
   - Check browser console for CORS errors

2. **Check environment variable:**
   - `REACT_APP_BACKEND_URL` is set correctly
   - Rebuild frontend after changing env vars

3. **Check backend URL:**
   - Verify backend is accessible
   - Test with curl or browser

### **Database Connection Issues:**

1. **Check MongoDB Atlas:**
   - IP whitelist includes `0.0.0.0/0`
   - Database user has correct permissions
   - Connection string is correct

2. **Check environment variable:**
   - `MONGO_URL` is set correctly
   - Password is URL-encoded if it contains special characters

---

## 📊 **Cost Comparison (Free Tiers)**

| Platform | Monthly Cost | Limitations |
|----------|-------------|-------------|
| **Railway** | $0 (with $5 credit) | Limited by credit |
| **Render** | $0 | 15 min spin-down |
| **Vercel** | $0 | 100GB bandwidth |
| **MongoDB Atlas** | $0 | 512MB storage |
| **Total** | **$0/month** | See limitations above |

---

## 🎯 **Recommended Setup**

**For Best Experience:**
- **Backend:** Railway (easiest, no spin-down)
- **Frontend:** Vercel (fast CDN, instant deploys)
- **Database:** MongoDB Atlas (free tier)

**For Simplicity:**
- **Everything:** Railway (backend + frontend)
- **Database:** MongoDB Atlas

---

## 📚 **Additional Resources**

- [Railway Documentation](https://docs.railway.app/)
- [Render Documentation](https://render.com/docs)
- [Vercel Documentation](https://vercel.com/docs)
- [MongoDB Atlas Setup](https://docs.atlas.mongodb.com/getting-started/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)

---

## ✅ **Quick Start Summary**

1. **Push code to GitHub**
2. **Set up MongoDB Atlas** (free)
3. **Deploy backend to Railway** (free tier)
4. **Deploy frontend to Railway or Vercel** (free tier)
5. **Set environment variables**
6. **Test and verify**

**Total Cost: $0/month** 🎉

---

**Ready to deploy?** Start with Railway - it's the easiest option for full-stack deployment!


