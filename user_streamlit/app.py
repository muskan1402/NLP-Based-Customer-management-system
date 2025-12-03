# user_streamlit/app.py
import streamlit as st
import requests
import os

API = "http://localhost:8000/api"

st.set_page_config(page_title="Reviews - User", layout="centered")

st.title("Submit a Review")

if 'token' not in st.session_state:
    st.session_state['token'] = None
    st.session_state['username'] = None

tab = st.sidebar.radio("Choose", ["Login", "Register", "Submit Review"])

if tab == "Register":
    st.subheader("Create an account")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        if not username or not email or not password:
            st.error("Fill all fields")
        else:
            r = requests.post(f"{API}/register", json={"username": username, "email": email, "password": password})
            if r.status_code == 200:
                st.success("Registered! Now login.")
            else:
                try:
                    st.error(r.json().get("detail", "Registration failed"))
                except ValueError:
                    st.error(f"Registration failed: {r.text or 'No response from server'}")


if tab == "Login":
    st.subheader("Login")
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login"):
        if not username or not password:
            st.error("Provide credentials")
        else:
            r = requests.post(f"{API}/login", json={"username": username, "password": password})
            if r.status_code == 200:
                token = r.json().get("access_token")
                st.session_state['token'] = token
                st.session_state['username'] = username
                st.success("Logged in")
            else:
                st.error("Login failed")

if tab == "Submit Review":
    if not st.session_state.get('token'):
        st.warning("You must login first (use sidebar -> Login)")
    else:
        st.subheader("Write your review")
        text = st.text_area("Your review text")
        if st.button("Submit Review"):
            if not text.strip():
                st.error("Please write a review")
            else:
                headers = {"Authorization": f"Bearer {st.session_state['token']}"}
                r = requests.post(f"{API}/reviews", json={"summary": text}, headers=headers)
                if r.status_code == 200:
                    st.success("Review submitted")
                else:
                    st.error(f"Failed: {r.status_code} {r.text}")
