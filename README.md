# Platable Streamlit Dashboard

Minimal, brand-aligned Streamlit dashboard for Platable (Company / Vendor / AM views), ready to run locally or deploy via Streamlit Community Cloud.

## Quick start (local)

```bash
git clone <your-repo-url>.git
cd platable-streamlit-dashboard
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this folder to GitHub.
2. Go to https://streamlit.io/cloud and create a new app.
3. Select your repo, branch `main`, and file `app.py`.
4. (Optional) Set secrets in **⚙️ Settings → Secrets** if you later add database creds.

## Repo structure

```
platable-streamlit-dashboard/
├─ app.py
├─ requirements.txt
├─ .gitignore
└─ .streamlit/
   └─ config.toml
```

## Brand theme

- Primary: `#E83D57`
- Text: `#111827`
- Secondary background: `#F7F7FA`

Update these in `.streamlit/config.toml` and in `app.py` (the `BRAND_PRIMARY` variable) if you need to tweak the hue.

---

© 2025 Platable. MIT license optional; add a LICENSE file if you want.
