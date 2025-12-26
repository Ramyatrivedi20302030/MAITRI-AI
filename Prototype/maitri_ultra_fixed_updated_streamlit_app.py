# MAITRI Ultra – Fixed & Updated Streamlit App
# Key fixes:
# - Syntax errors (@st.cache_resource-)
# - Password hashing/verification with salt
# - SQLite trigger order bug
# - GPU/CPU-safe model loading
# - Streamlit caching correctness
# - Security & stability improvements

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import datetime
import json
import sqlite3
import pandas as pd
import numpy as np
from io import BytesIO
import base64
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import hashlib, os, zipfile, io
from cryptography.fernet import Fernet
from datetime import timedelta
from scipy import stats
import torch

# ---------------- CONFIG ----------------
st.set_page_config(page_title="MAITRI Ultra", page_icon="🚀", layout="wide")

# ---------------- SECURITY ----------------
ENCRYPTION_KEY = os.getenv("MAITRI_ENCRYPT_KEY")
if not ENCRYPTION_KEY:
    ENCRYPTION_KEY = Fernet.generate_key().decode()
cipher_suite = Fernet(ENCRYPTION_KEY.encode())

# ---------------- DATABASE ----------------
@st.cache_resource
def init_db():
    conn = sqlite3.connect("maitri_ultra.db", check_same_thread=False)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            password_hash TEXT,
            salt BLOB,
            role TEXT DEFAULT 'user',
            created_at TEXT,
            last_login TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            psych INTEGER,
            phys INTEGER,
            overall REAL,
            text_encrypted BLOB,
            sentiment TEXT,
            confidence REAL,
            sleep_hours REAL,
            nutrition_notes TEXT,
            exercise_reps INTEGER,
            user_id TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_message_encrypted BLOB,
            ai_response TEXT,
            sentiment TEXT,
            confidence REAL,
            user_id TEXT,
            session_id TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            goal_text TEXT,
            progress REAL,
            target_date TEXT,
            created_date TEXT,
            completed INTEGER DEFAULT 0,
            priority INTEGER DEFAULT 1,
            user_id TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT,
            action TEXT,
            timestamp TEXT,
            user_id TEXT
        )
    """)

    c.execute("""
        CREATE TRIGGER IF NOT EXISTS audit_history_insert
        AFTER INSERT ON history
        BEGIN
            INSERT INTO audit_log (table_name, action, timestamp, user_id)
            VALUES ('history', 'INSERT', datetime('now'), NEW.user_id);
        END;
    """)

    conn.commit()
    return conn

conn = init_db()

# ---------------- MODELS ----------------
@st.cache_resource
def load_sentiment_model():
    name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    return pipeline("sentiment-analysis", model=name)

@st.cache_resource
def load_generator_model():
    name = "gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    tok.pad_token = tok.eos_token
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-generation", model=model, tokenizer=tok, device=device)

sentiment_pipeline = load_sentiment_model()
generator_pipeline = load_generator_model()

# ---------------- AUTH ----------------
def hash_password(password: str):
    salt = os.urandom(16)
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 120000)
    return pwd_hash, salt

def verify_password(password, stored_hash, salt):
    return stored_hash == hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 120000)

# ---------------- ENCRYPTION ----------------
def encrypt_text(text):
    return cipher_suite.encrypt(text.encode()) if text else None

def decrypt_text(blob):
    if not blob:
        return ""
    try:
        return cipher_suite.decrypt(blob).decode()
    except Exception:
        return "[Decryption failed]"

# ---------------- ANALYTICS ----------------
def detect_anomalies(df):
    if len(df) < 5:
        return []
    X = df[['psych', 'phys', 'overall']].fillna(0)
    X = StandardScaler().fit_transform(X)
    preds = IsolationForest(contamination=0.15, random_state=42).fit_predict(X)
    return df.index[preds == -1].tolist()

def predict_trend(df, days=7):
    if len(df) < 5:
        return None
    df = df.sort_values('date')
    x = np.arange(len(df)).reshape(-1, 1)
    y = df['overall'].values
    model = LinearRegression().fit(x, y)
    future = np.arange(len(df), len(df)+days).reshape(-1, 1)
    return model.predict(future)

# ---------------- UI ----------------
st.title("🚀 MAITRI Ultra – Fixed & Stable")
st.markdown("Secure • Predictive • Mission-Ready")

if 'user' not in st.session_state:
    st.session_state.user = None

if not st.session_state.user:
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            c = conn.cursor()
            c.execute("SELECT password_hash, salt FROM users WHERE user_id=?", (u,))
            r = c.fetchone()
            if r and verify_password(p, r[0], r[1]):
                st.session_state.user = u
                st.success("Logged in")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        u = st.text_input("New username")
        p = st.text_input("New password", type="password")
        if st.button("Register"):
            h, s = hash_password(p)
            try:
                conn.execute("INSERT INTO users VALUES (?,?,?,?,?,?)",
                             (u, h, s, 'user', datetime.datetime.now().isoformat(), None))
                conn.commit()
                st.success("Registered")
            except sqlite3.IntegrityError:
                st.error("User exists")

else:
    st.success(f"Welcome {st.session_state.user}")
    msg = st.text_area("Talk to MAITRI")
    if st.button("Send") and msg:
        sent = sentiment_pipeline(msg)[0]
        reply = generator_pipeline(msg, max_length=120)[0]['generated_text']
        st.write("**MAITRI:**", reply)
