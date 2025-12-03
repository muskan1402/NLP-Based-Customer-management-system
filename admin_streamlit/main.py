"""
admin_streamlit/main.py

Admin-only Streamlit dashboard for ReviewScope.
- Requires admin login (calls backend /api/login).
- Loads local roberta_model only after successful login (local_files_only=True).
- Fetches reviews from backend /api/reviews and shows analytics.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Transformers imports done lazily (only when model is used)
from scipy.special import softmax

st.set_page_config(page_title="Admin Dashboard (Protected)", layout="wide")

# ---- CONFIG ----
API_BASE = "http://127.0.0.1:8000/api"  # backend base url
# model folder relative to this file (adjust if your roberta_model is elsewhere)
REL_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "roberta_model"))
# sample CSV path relative to repo root (admin_streamlit/main.py -> ../data/sample.csv)
SAMPLE_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample.csv"))

# ---- HELPERS ----
def admin_login_request(username: str, password: str, timeout=6):
    """Call backend /api/login and return token or raise."""
    resp = requests.post(f"{API_BASE}/login", json={"username": username, "password": password}, timeout=timeout)
    return resp

@st.cache_resource
def load_local_model(abs_path):
    """Load tokenizer+model from local folder only. Returns (tokenizer, model) or (None, None)."""
    try:
        if not os.path.isdir(abs_path):
            st.warning(f"Model folder not found at: {abs_path}")
            return None, None
        # Import transformers lazily
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(abs_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(abs_path, local_files_only=True)
        return tokenizer, model
    except Exception as e:
        st.warning(f"Failed to load model from {abs_path}: {e}")
        return None, None

def fetch_reviews_from_api(timeout=6):
    """Fetch reviews from backend /api/reviews. Returns DataFrame (may be empty)."""
    try:
        r = requests.get(f"{API_BASE}/reviews", timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                return pd.DataFrame(data)
            else:
                st.warning("Unexpected /api/reviews response format.")
                return pd.DataFrame()
        else:
            st.warning(f"Failed to fetch reviews: {r.status_code} {r.text}")
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not reach backend /api/reviews: {e}")
        return pd.DataFrame()

def safe_int_max(series):
    """Return int max of series or 0 if not possible."""
    if series is None or len(series) == 0:
        return 0
    try:
        if series.dtype.kind in "fiu":  # numeric
            if series.isna().all():
                return 0
            return int(np.nanmax(series))
        else:
            # non-numeric -> try cast
            s = pd.to_numeric(series, errors='coerce')
            if s.isna().all():
                return 0
            return int(np.nanmax(s))
    except Exception:
        return 0

def normalize_reviews_df(df: pd.DataFrame):
    """Normalize incoming reviews df to columns: Id, Summary, Timestamp, Sentiment, Category (optional)"""
    if df is None or df.empty:
        return pd.DataFrame(columns=['Id','Summary','Timestamp','Sentiment'])
    d = df.copy()
    # Summary
    if 'Summary' not in d.columns:
        if 'summary' in d.columns:
            d = d.rename(columns={'summary':'Summary'})
        elif 'text' in d.columns:
            d = d.rename(columns={'text':'Summary'})
    # Timestamp
    if 'Timestamp' in d.columns:
        d['Timestamp'] = pd.to_datetime(d['Timestamp'], errors='coerce')
    elif 'timestamp' in d.columns:
        d['Timestamp'] = pd.to_datetime(d['timestamp'], errors='coerce')
    else:
        d['Timestamp'] = pd.to_datetime('now')
    # Sentiment
    if 'Sentiment' not in d.columns:
        if 'sentiment' in d.columns:
            d = d.rename(columns={'sentiment':'Sentiment'})
        else:
            d['Sentiment'] = 'unknown'
    d['Sentiment'] = d['Sentiment'].fillna('unknown')
    # Id
    if 'Id' not in d.columns:
        d['Id'] = np.nan
    # Keep only needed columns and fill missing
    out = d[['Id','Summary','Timestamp','Sentiment']].copy()
    out['Summary'] = out['Summary'].astype(str)
    return out

# ---- SESSION KEYS ----
if 'admin_token' not in st.session_state:
    st.session_state['admin_token'] = None
if 'admin_username' not in st.session_state:
    st.session_state['admin_username'] = None

# ---- LOGIN UI (sidebar) ----
st.sidebar.title("Admin Login (required)")
username = st.sidebar.text_input("Username", key="admin_username_input")
password = st.sidebar.text_input("Password", type="password", key="admin_password_input")
if st.sidebar.button("Login"):
    if not username or not password:
        st.sidebar.error("Please enter username and password.")
    else:
        try:
            resp = admin_login_request(username, password)
            if resp.status_code == 200:
                token = resp.json().get("access_token")
                st.session_state['admin_token'] = token
                st.session_state['admin_username'] = username
                st.sidebar.success("Logged in.")
            else:
                st.sidebar.error(f"Login failed: {resp.text}")
        except Exception as e:
            st.sidebar.error(f"Login error: {e}")

# If not logged in, show message and stop
if not st.session_state['admin_token']:
    st.title("Admin Dashboard — Login required")
    st.info("Please login from the sidebar to access analytics.")
    # gentle note where to go if backend unreachable
    st.caption("Backend must be running at http://127.0.0.1:8000")
    st.stop()

# ---- At this point - admin is authenticated ----
admin_token = st.session_state['admin_token']
st.title("Admin Analytics Dashboard")
st.markdown(f"Logged in as **{st.session_state.get('admin_username', '')}**")

# ---- Load the model lazily (after login) ----
tokenizer, model = load_local_model(REL_MODEL_DIR)

# ---- Load sample csv (if present) ----
if os.path.exists(SAMPLE_CSV_PATH):
    try:
        sample_df = pd.read_csv(SAMPLE_CSV_PATH)
    except Exception as e:
        st.warning(f"Could not read sample CSV at {SAMPLE_CSV_PATH}: {e}")
        sample_df = pd.DataFrame()
else:
    sample_df = pd.DataFrame()
    st.info(f"Sample CSV not found at {SAMPLE_CSV_PATH}. Dashboard will still show backend review")

# Normalize sample columns
if not sample_df.empty:
    # Rename possible matches
    for wanted in ['Summary','Timestamp','Sentiment','Id','ProductId']:
        if wanted not in sample_df.columns:
            # try case-insensitive match
            for c in sample_df.columns:
                if c.lower() == wanted.lower():
                    sample_df = sample_df.rename(columns={c: wanted})
                    break
    # ensure Timestamp dtype
    if 'Timestamp' in sample_df.columns:
        sample_df['Timestamp'] = pd.to_datetime(sample_df['Timestamp'], errors='coerce')
    else:
        sample_df['Timestamp'] = pd.to_datetime('now')
    if 'Id' not in sample_df.columns:
        sample_df['Id'] = range(1, len(sample_df) + 1)

# ---- Fetch API reviews and normalize/merge ----
api_df = fetch_reviews_from_api()
api_df = normalize_reviews_df(api_df)

# safe id generation for api_df
last_id = safe_int_max(sample_df['Id']) if not sample_df.empty else 0
api_df_len = len(api_df)
if api_df_len > 0:
    api_df['Id'] = list(range(last_id + 1, last_id + 1 + api_df_len))

# session-local user-submitted reviews (preview)
if 'user_reviews_df' not in st.session_state:
    st.session_state['user_reviews_df'] = pd.DataFrame(columns=['Id','Summary','Timestamp','Sentiment','Score'])

# Combine datasets (sample + api + session submissions)
parts = []
if not sample_df.empty:
    parts.append(sample_df[['Id','Summary','Timestamp','Sentiment']])
if not api_df.empty:
    parts.append(api_df[['Id','Summary','Timestamp','Sentiment']])
if not st.session_state['user_reviews_df'].empty:
    parts.append(st.session_state['user_reviews_df'][['Id','Summary','Timestamp','Sentiment']])
if parts:
    combined = pd.concat(parts, ignore_index=True, sort=False)
else:
    combined = pd.DataFrame(columns=['Id','Summary','Timestamp','Sentiment'])

# Normalize combined
if 'Timestamp' in combined.columns:
    combined['Timestamp'] = pd.to_datetime(combined['Timestamp'], errors='coerce')
else:
    combined['Timestamp'] = pd.to_datetime('now')
combined = combined.dropna(subset=['Summary']).reset_index(drop=True)
combined['Sentiment'] = combined['Sentiment'].fillna('unknown')

# If no data, show message
if combined.empty:
    st.info("No reviews available yet. Ask users to submit reviews through the User Panel.")
    st.stop()

# ---- Build charts ----
st.subheader("Sentiment Overview")
sent_counts = combined['Sentiment'].value_counts()

# Pie
fig1, ax1 = plt.subplots(figsize=(6,4))
ax1.pie(sent_counts, labels=sent_counts.index, autopct='%1.1f%%', startangle=140)
ax1.set_title("Sentiment Distribution")
st.pyplot(fig1)

# Bar
fig2, ax2 = plt.subplots(figsize=(6,4))
sns.countplot(x='Sentiment', data=combined, order=sent_counts.index, ax=ax2)
ax2.set_title("Count per Sentiment")
st.pyplot(fig2)

# Category distribution (best-effort)
st.subheader("Category / Sentiment distribution")
if 'ProductId' in sample_df.columns:
    # create simple category mapping for sample rows, API rows -> Others
    categories = ['Beverages', 'Snacks', 'Pantry', 'Household', 'Health', 'Others']
    def cat_of_pid(x):
        try:
            return categories[hash(x) % len(categories)]
        except Exception:
            return 'Others'
    combined['Category'] = combined.get('ProductId', pd.Series([None]*len(combined))).apply(lambda x: cat_of_pid(x) if pd.notna(x) else 'Others')
else:
    combined['Category'] = 'Others'

try:
    pivot = pd.pivot_table(combined, values='Id', index='Category', columns='Sentiment', aggfunc='count').fillna(0)
    fig3, ax3 = plt.subplots(figsize=(8,4))
    neg = pivot.get('negative', 0)
    neu = pivot.get('neutral', 0)
    pos = pivot.get('positive', 0)
    ax3.bar(pivot.index, neg, label='Negative', color='red')
    ax3.bar(pivot.index, neu, bottom=neg, label='Neutral', color='orange')
    ax3.bar(pivot.index, pos, bottom=(neg+neu), label='Positive', color='green')
    ax3.set_title("Sentiment by Category (stacked)")
    ax3.set_ylabel("Count")
    ax3.legend()
    st.pyplot(fig3)
except Exception:
    st.info("Not enough category data to build stacked chart.")

# Monthly trends (if timestamps ok)
st.subheader("Monthly Trend")
try:
    if not combined['Timestamp'].isna().all():
        # monthly counts: group by period
        monthly = combined.groupby(combined['Timestamp'].dt.to_period('M')).count()['Id']
        monthly.index = monthly.index.to_timestamp()
        fig4, ax4 = plt.subplots(figsize=(10,3))
        ax4.plot(monthly.index, monthly.values, marker='o')
        ax4.set_title("Monthly Review Count")
        ax4.set_xlabel("Month")
        ax4.set_ylabel("Count")
        ax4.grid(True)
        st.pyplot(fig4)
except Exception:
    st.info("Not enough time-series data for trend plot.")

# Wordclouds
st.subheader("Word Clouds by Sentiment")
wc_cols = st.columns(3)
for i, sentiment_label in enumerate(['positive','neutral','negative']):
    df_sent = combined[combined['Sentiment'] == sentiment_label]
    text = " ".join(df_sent['Summary'].astype(str))
    if text.strip():
        wc = WordCloud(width=600, height=300).generate(text)
        fig_wc, ax_wc = plt.subplots(figsize=(6,3))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        ax_wc.set_title(sentiment_label.capitalize())
        wc_cols[i].pyplot(fig_wc)
    else:
        wc_cols[i].write(f"No {sentiment_label} reviews yet.")

# Recent reviews table
st.subheader("Recent Reviews (most recent first)")
st.dataframe(combined.sort_values('Timestamp', ascending=False).head(100))

# Quick admin-only inference (if model loaded)
st.markdown("---")
st.subheader("Admin: Quick text inference (local model)")
if tokenizer is None or model is None:
    st.warning("Local model not loaded or not available. Place your model folder in 'roberta_model' at repo root.")
else:
    txt = st.text_area("Enter text to analyze")
    if st.button("Analyze text"):
        if not txt.strip():
            st.error("Enter some text first.")
        else:
            enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=256)
            out = model(**enc)
            scores = softmax(out.logits[0].detach().numpy())
            labels = ['negative','neutral','positive']
            idx = int(np.argmax(scores))
            st.success(f"Predicted sentiment: **{labels[idx]}** (score={float(scores[idx]):.3f})")

# Footer note
st.caption("Admin Dashboard — shows reviews fetched from backend and sample CSV.")
