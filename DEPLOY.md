# Deploying to Streamlit Community Cloud

This repo is already set up for it: `.streamlit/config.toml` has a theme matching the
project's palette, `requirements.txt` uses `tensorflow-cpu` (not the full GPU package,
which is too heavy for the free tier), `runtime.txt` pins Python 3.11, and
`real_data.py` auto-downloads its dataset at runtime — so the repo doesn't need to
carry a 3.7MB CSV.

## 1. Push this repo to GitHub

```bash
cd shipping_cost_pipeline
# (git is already initialized with an initial commit)
git remote add origin https://github.com/<your-username>/shipping-cost-pipeline.git
git branch -M main
git push -u origin main
```

If you don't have a GitHub repo yet: go to github.com → New repository → name it
(e.g. `shipping-cost-pipeline`) → leave it empty (no README/license, this repo already
has those) → create → copy the URL it gives you into the `git remote add` command above.

## 2. Deploy on Streamlit Community Cloud

1. Go to **share.streamlit.io** and sign in with GitHub.
2. Click **New app**.
3. Pick your repo, branch `main`, and main file path `streamlit_app.py`.
4. Click **Deploy**.

First deploy takes a few minutes (installing `tensorflow-cpu`, `lightgbm`, `xgboost`,
etc.). Subsequent restarts are fast since dependencies are cached.

## 3. What to expect on first run

- The synthetic-data path works immediately — no external calls needed.
- The **"Use real data"** toggle triggers `real_data.py`'s auto-download from GitHub on
  first use (a few seconds, then cached in the app's filesystem for the session).
- **Faithful mode** (30-fold CV, 500-epoch DNN) will likely time out or feel painfully
  slow on the free tier's shared CPU — it's meant for local runs with more compute. Demo
  mode (the default) is what you want for a live portfolio link.

## 4. Put the link in your portfolio

Once deployed you'll get a URL like `https://<your-app-name>.streamlit.app`. That's a
better portfolio link than "clone this repo and run it" — it demonstrates the project
end-to-end with zero setup for whoever's looking at it (e.g. a recruiter or interviewer).

Worth pairing the link with the one-line pitch: *"A Claude Code skill I built that
converts tabular ML papers into runnable pipelines; here's it applied to a shipping-cost
prediction paper, validated against both synthetic and real logistics data."* That
framing is more accurate — and more interesting — than claiming a paper reproduction.
