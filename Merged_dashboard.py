# merged_app.py
"""
Single Streamlit app combining User panel + Admin dashboard.
Place this file at repo root and run: `streamlit run merged_app.py`
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import matplotlib.pyplot as plt

# Optional libs used by admin dashboard; if you don't have them it's fine
try:
    import seaborn as sns
except Exception:
    sns = None
try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None
from scipy.special import softmax

st.set_page_config(page_title="Review analysis and automation - User & Admin", layout="wide")

# ---------------- CONFIG ----------------
API_BASE = os.environ.get("REVSCOPE_API", "http://127.0.0.1:8000/api")
REL_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "roberta_model"))
SAMPLE_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "sample.csv"))

# ----------------- Styling -----------------
def local_css():
    st.markdown(
        """
        <style>
        /* ---------------- Main app ---------------- */
        .stApp {
            background: linear-gradient(120deg, #e9eef6 0%, #c6d2e6 100%) !important;
            color: #111 !important;
        }

        /* --------------- Header ---------------- */
        .header {
            background: linear-gradient(90deg,blue 0%, pink 100%);
            padding: 18px 22px;
            border-radius: 12px;
            color: blue !important;
        }
        .header h1 { color:#fff !important; }

        /* --------------- Card ------------------ */
        .card {
            background: #ffffffdd !important;
            border-radius: 12px !important;
            padding: 14px !important;
            box-shadow: 0 6px 18px rgba(0,0,0,0.08) !important;
            color: pink !important;
        }

        /* --------------- Buttons / Inputs --------------- */
        .stButton>button {
            background: #1f5d9c !important;
            color: blue !important;
            border-radius: 8px !important;
            padding: 8px 14px !important;
            border: none !important;
        }
        .stButton>button:hover { background: #174a78 !important; }
        input, textarea, .stTextInput input, .stTextArea textarea {
            color: #111 !important;
            background: #fff !important;
        }

        /* --------------- Global text fallback --------------- */
        body, p, span, div, label, h1, h2, h3 {
            color: #111 !important;
        }

        /* ========== SAFE SIDEBAR STYLING ========== */
        /* Set a readable background for the sidebar container */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f7fbff 0%, #e6eef9 100%) !important;
            color: #111 !important;
            padding: 16px 12px 24px 12px !important;
        }

        /* Make all text inside sidebar visible */
        section[data-testid="stSidebar"] * {
            color: #111 !important;
            background: transparent !important;
            fill: #111 !important;
        }

        /* Sidebar headings, radio labels, links */
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .css-1lsmgbg {
            color: #111 !important;
            font-weight: 600 !important;
        }

        /* Ensure radio/checkbox labels (menu items) are readable */
        section[data-testid="stSidebar"] div[role="radiogroup"] label,
        section[data-testid="stSidebar"] label[for] {
            color: #111 !important;
        }

        /* Tweak the small text items in the sidebar */
        section[data-testid="stSidebar"] .stText, 
        section[data-testid="stSidebar"] .stMarkdown {
            color: #111 !important;
        }

        /* Keep any icons in sidebar visible */
        section[data-testid="stSidebar"] svg { fill: #111 !important; color: #111 !important; }

        /* Minor: keep dataframes text readable */
        .stDataFrame div {
            color: #111 !important;
        }

        /* Fallback button style inside sidebar */
        section[data-testid="stSidebar"] .stButton>button {
            background: #2d6ca3 !important;
            color: #fff !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# call styling
local_css()

# ----------------- Helpers -----------------
def api_post(path, json=None, headers=None, timeout=6):
    try:
        r = requests.post(f"{API_BASE.rstrip('/')}/{path.lstrip('/')}", json=json, headers=headers or {}, timeout=timeout)
        return r
    except Exception as e:
        return None

def api_get(path, headers=None, timeout=6):
    try:
        r = requests.get(f"{API_BASE.rstrip('/')}/{path.lstrip('/')}", headers=headers or {}, timeout=timeout)
        return r
    except Exception as e:
        return None

# normalize review df (adapted from your admin script)
def normalize_reviews_df(df: pd.DataFrame):
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
    if 'Id' not in d.columns:
        d['Id'] = np.nan
    out = d[['Id','Summary','Timestamp','Sentiment']].copy()
    out['Summary'] = out['Summary'].astype(str)
    return out

@st.cache_resource
def load_local_model(abs_path):
    try:
        if not os.path.isdir(abs_path):
            return None, None
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(abs_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(abs_path, local_files_only=True)
        return tokenizer, model
    except Exception as e:
        return None, None

# ----------------- Session state defaults -----------------
if 'role' not in st.session_state:
    st.session_state['role'] = None   # None / "user" / "admin"
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'token' not in st.session_state:
    st.session_state['token'] = None
if 'admin_token' not in st.session_state:
    st.session_state['admin_token'] = None

# ----------------- Layout -----------------
with st.container():
    st.markdown(
        """
        <div class="header">
            <h1 style="margin:0;">Review analysis and automation</h1>
            <div style="margin-top:6px;">Universal Dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns([3,1])
    with cols[1]:
        if st.session_state['role'] == 'admin':
            st.markdown(f"<span class='badge'>Admin</span><div class='subtle'>Signed in as <b>{st.session_state['username']}</b></div>", unsafe_allow_html=True)
            if st.button("Logout admin"):
                st.session_state['role'] = None
                st.session_state['admin_token'] = None
                st.session_state['username'] = None
                st.experimental_rerun()
        elif st.session_state['role'] == 'user':
            st.markdown(f"<span class='badge'>User</span><div class='subtle'>Signed in as <b>{st.session_state['username']}</b></div>", unsafe_allow_html=True)
            if st.button("Logout"):
                st.session_state['role'] = None
                st.session_state['token'] = None
                st.session_state['username'] = None
                st.experimental_rerun()
        else:
            st.markdown("<div class='subtle'>Not signed in</div>", unsafe_allow_html=True)

st.markdown("---")

# ----------------- Sidebar navigation -----------------
menu = st.sidebar.radio("Navigate", ["Home", "User Panel", "Admin Panel", "About"], index=1 if st.session_state['role']=='user' else 2 if st.session_state['role']=='admin' else 0)

# --------------- Home -------------------
if menu == "Home":
    st.subheader("Welcome")
    st.write("Use the sidebar to go to the User Panel or Admin Panel.")
    st.info("Running backend API at `http://127.0.0.1:8000` + folder is present for admin inference.")
    # optional recent reviews preview
    try:
        r = api_get("reviews")
        if r and r.status_code == 200:
            df = pd.DataFrame(r.json())
            if not df.empty:
                dfn = normalize_reviews_df(df)
                st.markdown("### Recent reviews (from backend)")
                st.dataframe(dfn.sort_values('Timestamp', ascending=False).head(10))
    except Exception:
        pass

# --------------- User Panel -------------------
if menu == "User Panel":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("User Panel — Register / Login / Submit review")
    st.write("Simple flow: Register → Login → Submit review")
    st.markdown('</div>', unsafe_allow_html=True)

    tab = st.radio("Choose", ["Login", "Register", "Submit Review"], index=0)

    if tab == "Register":
        st.subheader("Create an account")
        r_user = st.text_input("Username", key="reg_user")
        r_email = st.text_input("Email", key="reg_email")
        r_pass = st.text_input("Password", type="password", key="reg_pass")
        if st.button("Register", key="reg_btn"):
            if not r_user or not r_email or not r_pass:
                st.error("Fill all fields")
            else:
                r = api_post("register", json={"username": r_user, "email": r_email, "password": r_pass})
                if r is None:
                    st.error("Could not reach backend.")
                elif r.status_code == 200:
                    st.success("Registered! Please login.")
                else:
                    try:
                        st.error(r.json().get("detail","Registration failed"))
                    except Exception:
                        st.error(f"Registration failed: {r.status_code}")

    if tab == "Login":
        st.subheader("Login")
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", key="login_btn"):
            if not u or not p:
                st.error("Provide credentials")
            else:
                r = api_post("login", json={"username": u, "password": p})
                if r is None:
                    st.error("Could not reach backend.")
                elif r.status_code == 200:
                    token = r.json().get("access_token")
                    st.session_state['token'] = token
                    st.session_state['username'] = u
                    st.session_state['role'] = 'user'
                    st.success("Logged in")
                else:
                    st.error("Login failed")

    if tab == "Submit Review":
        if not st.session_state.get('token'):
            st.warning("You must login first (use Login tab)")
        else:
            st.subheader("Write your review")
            text = st.text_area("Your review text", height=140)
            if st.button("Submit Review", key="submit_review"):
                if not text.strip():
                    st.error("Please write a review")
                else:
                    headers = {"Authorization": f"Bearer {st.session_state['token']}"}
                    r = api_post("reviews", json={"summary": text}, headers=headers)
                    if r is None:
                        st.error("Could not reach backend.")
                    elif r.status_code == 200:
                        st.success("Review submitted")
                    else:
                        st.error(f"Failed: {r.status_code} {r.text}")

# --------------- Admin Panel -------------------
if menu == "Admin Panel":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Admin Panel — Protected dashboard")
    st.markdown('</div>', unsafe_allow_html=True)

    # If not logged in as admin: show admin login
    if st.session_state.get('role') != 'admin' or not st.session_state.get('admin_token'):
        st.subheader("Admin login")
        a_user = st.text_input("Admin username", key="admin_user")
        a_pass = st.text_input("Admin password", type="password", key="admin_pass")
        if st.button("Login as admin", key="admin_login_btn"):
            if not a_user or not a_pass:
                st.error("Enter admin credentials")
            else:
                r = api_post("login", json={"username": a_user, "password": a_pass})
                if r is None:
                    st.error("Could not reach backend.")
                elif r.status_code == 200:
                    token = r.json().get("access_token")
                    st.session_state['admin_token'] = token
                    st.session_state['username'] = a_user
                    st.session_state['role'] = 'admin'
                    st.success("Admin logged in")
                else:
                    st.error("Admin login failed")
    else:
        # ----------------- Admin authenticated: show dashboard -----------------
        st.markdown("### Admin analytics")
        st.write("Fetching reviews and showing analytics for Admin.")

        # Load sample csv
        sample_df = pd.DataFrame()
        if os.path.exists(SAMPLE_CSV_PATH):
            try:
                sample_df = pd.read_csv(SAMPLE_CSV_PATH)
            except Exception as e:
                st.warning(f"Could not read sample CSV: {e}")

        # fetch reviews from API
        r = api_get("reviews", headers={"Authorization": f"Bearer {st.session_state['admin_token']}"})
        api_df = pd.DataFrame()
        if r and r.status_code == 200:
            try:
                api_df = pd.DataFrame(r.json())
            except Exception:
                api_df = pd.DataFrame()
        api_df = normalize_reviews_df(api_df)

        # safe id for api rows
        if not sample_df.empty and 'Id' in sample_df.columns:
            last_id = int(sample_df['Id'].max() if not sample_df['Id'].isna().all() else 0)
        elif not sample_df.empty:
            last_id = len(sample_df)
        else:
            last_id = 0
        if len(api_df) > 0:
            api_df['Id'] = list(range(last_id + 1, last_id + 1 + len(api_df)))

        # session user-submitted preview (optional)
        if 'user_reviews_df' not in st.session_state:
            st.session_state['user_reviews_df'] = pd.DataFrame(columns=['Id','Summary','Timestamp','Sentiment','Score'])

        # combine
        parts = []
        if not sample_df.empty:
            # ensure minimal columns
            sample_df = sample_df.rename(columns={c:c for c in sample_df.columns})
            if 'Timestamp' in sample_df.columns:
                sample_df['Timestamp'] = pd.to_datetime(sample_df['Timestamp'], errors='coerce')
            parts.append(sample_df[['Id','Summary','Timestamp','Sentiment']] if 'Id' in sample_df.columns else sample_df[['Summary','Timestamp','Sentiment']])
        if not api_df.empty:
            parts.append(api_df[['Id','Summary','Timestamp','Sentiment']])
        if not st.session_state['user_reviews_df'].empty:
            parts.append(st.session_state['user_reviews_df'][['Id','Summary','Timestamp','Sentiment']])
        if parts:
            combined = pd.concat(parts, ignore_index=True, sort=False)
        else:
            combined = pd.DataFrame(columns=['Id','Summary','Timestamp','Sentiment'])
        if 'Timestamp' in combined.columns:
            combined['Timestamp'] = pd.to_datetime(combined['Timestamp'], errors='coerce')
        else:
            combined['Timestamp'] = pd.to_datetime('now')
        combined = combined.dropna(subset=['Summary']).reset_index(drop=True)
        combined['Sentiment'] = combined['Sentiment'].fillna('unknown')

        if combined.empty:
            st.info("No reviews available yet. Ask users to submit reviews through the User Panel.")
        else:
            # Sentiment overview
            st.subheader("Sentiment Overview")
            sent_counts = combined['Sentiment'].value_counts()
            # small two-column layout for charts + table
            c1, c2 = st.columns([1,1])
            with c1:
                fig1, ax1 = plt.subplots(figsize=(5,3))
                ax1.pie(sent_counts, labels=sent_counts.index, autopct='%1.1f%%', startangle=140)
                ax1.set_title("Sentiment Distribution")
                st.pyplot(fig1)
            with c2:
                fig2, ax2 = plt.subplots(figsize=(5,3))
                if sns is not None:
                    sns.countplot(x='Sentiment', data=combined, order=sent_counts.index, ax=ax2)
                else:
                    combined['Sentiment'].value_counts().plot(kind='bar', ax=ax2)
                ax2.set_title("Count per Sentiment")
                st.pyplot(fig2)

            # monthly trend
            st.subheader("Monthly Trend")
            try:
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

            # wordclouds (if available)
            if WordCloud is not None:
                st.subheader("Word clouds by sentiment")
                cols = st.columns(3)
                for i, lab in enumerate(['positive','neutral','negative']):
                    df_s = combined[combined['Sentiment'] == lab]
                    text = " ".join(df_s['Summary'].astype(str))
                    if text.strip():
                        wc = WordCloud(width=600, height=300).generate(text)
                        fig_wc, ax_wc = plt.subplots(figsize=(6,3))
                        ax_wc.imshow(wc, interpolation='bilinear')
                        ax_wc.axis('off')
                        ax_wc.set_title(lab.capitalize())
                        cols[i].pyplot(fig_wc)
                    else:
                        cols[i].write(f"No {lab} reviews yet.")
            else:
                st.warning("wordcloud lib not available - install `wordcloud` for word clouds")

            st.subheader("Recent reviews (most recent first)")
            st.dataframe(combined.sort_values('Timestamp', ascending=False).head(200))

        # Quick admin text inference (local model)
        st.markdown("---")
        st.subheader("Admin: Quick text inference (local model)")
        tokenizer, model = load_local_model(REL_MODEL_DIR)
        if tokenizer is None or model is None:
            st.warning("Local model not loaded or not available. Place 'roberta_model' folder in repo root.")
        else:
            txt = st.text_area("Enter text to analyze", key="admin_infer_text")
            if st.button("Analyze text (admin)", key="admin_infer_btn"):
                if not txt.strip():
                    st.error("Enter text first")
                else:
                    enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=256)
                    out = model(**enc)
                    scores = softmax(out.logits[0].detach().numpy())
                    labels = ['negative','neutral','positive']
                    idx = int(np.argmax(scores))
                    st.success(f"Predicted sentiment: **{labels[idx]}** (score={float(scores[idx]):.3f})")

# --------------- About -------------------
if menu == "About":
    st.subheader("About this app")
    st.write(
        """
        - A unified Streamlit application that integrates both User Panel and Admin Dashboard in a single interface.
        - Communicates with a FastAPI backend using secure REST APIs for authentication, review submission, and analytics.
        - Utilizes a fine-tuned RoBERTa transformer model to automatically classify user reviews into positive, neutral, or negative sentiments.
        - Provides an interactive Admin Analytics Dashboard with charts, trends, and word clouds, enhanced through custom inline CSS for a modern UI experience.
        """
    )
    st.markdown("**API base**")
    st.write(API_BASE)

