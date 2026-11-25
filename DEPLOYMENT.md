# Deployment Guide

## Method: Google Colab + ngrok (Free)

This is the simplest free way to make CineMatch publicly accessible.  
Google Colab runs the app on Google's servers — your PC is not involved.

---

## Steps

### 1. Open the notebook in Google Colab

Click this link:  
**https://colab.research.google.com/github/Abhinand0708/cinematch/blob/main/CineMatch_Deploy.ipynb**

Or open it from the repo:  
- Go to `CineMatch_Deploy.ipynb` in the GitHub repo
- Click **Open in Colab** badge

### 2. Run all cells

Click **Runtime → Run all** (or press `Ctrl+F9`).

The notebook will:
1. Install all dependencies (~2 min)
2. Clone the GitHub repo
3. Download model files from Hugging Face Hub (~2 min)
4. Set environment variables
5. Download NLTK data
6. Start Flask and open a public ngrok tunnel

### 3. Get your public URL

At the end of the last cell you will see:

```
🎬 CineMatch is LIVE!
   Public URL: https://xxxx.ngrok-free.app
```

Copy the `https://xxxx.ngrok-free.app` URL and share it with anyone.

### 4. Keep the tab open

The server runs as long as the Colab tab is open.  
Google Colab free sessions last up to **12 hours**.

When the session ends, just open the notebook again and run all cells — takes about 5 minutes.

---

## Notes

- The ngrok URL changes every time you restart the session (free tier limitation)
- When visitors open the link they may see an ngrok warning page — they click **Visit Site** to continue
- Google Colab gives 12GB RAM — more than enough for the app
- Your PC only needs a browser tab open — all processing happens on Google's servers

---

## For Permanent 24/7 Hosting

If you want the app always online without keeping Colab open, consider:

| Option | Cost | RAM | Notes |
|---|---|---|---|
| Oracle Cloud Free Tier | Free forever | 24GB ARM | Needs credit card for verification |
| Railway | $5/month | 8GB | Easiest setup, connects to GitHub |
| Render | $7/month | 2GB | Simple, good documentation |

---

## Model Files

Model and dataset files are stored on Hugging Face Hub (not in the GitHub repo):  
**https://huggingface.co/datasets/abhinand8386/cinematch-models**

The Colab notebook downloads them automatically.  
To download them locally, run:
```bash
python download_models.py
```
