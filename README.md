# ✦ Aura Check — Free Face Aura Scorer

A minimal, premium web app that scores your **aura** using MediaPipe + OpenCV.
No paid APIs. Runs locally or free on Render.com.

---

## 📁 Project Structure

```
aura-app/
├── backend/
│   ├── app.py            ← Flask server
│   ├── aura_engine.py    ← Scoring logic (MediaPipe + OpenCV)
│   └── requirements.txt
├── frontend/
│   ├── index.html        ← Upload page
│   └── result.html       ← Animated result card
├── render.yaml           ← One-click Render.com deploy
└── README.md
```

---

## 🏃 Run Locally (Your Computer)

### Step 1 — Install Python dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 2 — Start the server
```bash
python app.py
```

### Step 3 — Open in browser
```
http://localhost:5000
```

---

## 🌐 Deploy FREE on Render.com (Online)

### Step 1 — Push to GitHub
1. Create a free account at https://github.com
2. Create a new repository called `aura-check`
3. Upload all files (drag & drop on GitHub works)

### Step 2 — Deploy on Render
1. Go to https://render.com — create a free account
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account
4. Select your `aura-check` repository
5. Render auto-detects `render.yaml` — click **Deploy**
6. Wait ~3 minutes for the build

### Step 3 — Get your free URL
```
https://aura-check.onrender.com   ← share this with anyone!
```

> ⚠️ Free Render apps sleep after 15 min of inactivity.
> First visit after sleep takes ~30 seconds to wake up. That's normal.

---

## 🧠 How Aura Score is Calculated

| Metric           | Method                                     | Weight |
|------------------|--------------------------------------------|--------|
| Face Symmetry    | MediaPipe 468-point landmark mirroring     | 25%    |
| Skin Glow        | LAB brightness + Laplacian smoothness      | 25%    |
| Eye Intensity    | Eye aspect ratio + iris/sclera contrast    | 25%    |
| Jawline          | Canny edge density along chin contour      | 25%    |

### Aura Tiers
| Score  | Tier          |
|--------|---------------|
| 90–100 | Godly Aura ⚡  |
| 76–89  | Elite Aura 👑  |
| 61–75  | Rare Aura 💜   |
| 41–60  | Mid Aura ✨    |
| 0–40   | Developing 🌱  |

---

## 🛠 Tech Stack

- **Backend**: Python · Flask · MediaPipe · OpenCV · NumPy
- **Frontend**: Pure HTML · CSS · Vanilla JS (no frameworks)
- **Hosting**: Render.com (free tier)

---

## ❓ Troubleshooting

**"No face detected"** → Use a clear, front-facing photo with good lighting.

**Slow first load on Render** → Normal — free tier sleeps. Wait 30s.

**MediaPipe install error** → Make sure Python 3.8–3.11 is used (not 3.12+).
