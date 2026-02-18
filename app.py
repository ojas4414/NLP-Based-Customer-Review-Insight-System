import streamlit as st
import joblib
import numpy as np
import sys

# ---------------- PATH SETUP ----------------
sys.path.append("src")
from preprocessing import clean_text

# ---------------- LOAD MODEL & VECTORIZER ----------------
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/sentiment_svm_model.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Review Sentiment Analyzer",
    layout="centered",
)

# ---------------- CYBERPUNK UI THEME (UI ONLY) ----------------
st.markdown(
    """
    <style>
    /* ===== BACKGROUND ===== */
    .stApp {
        background:
            radial-gradient(circle at 20% 20%, rgba(0, 255, 255, 0.18), transparent 35%),
            radial-gradient(circle at 80% 30%, rgba(255, 0, 255, 0.14), transparent 40%),
            radial-gradient(circle at 50% 85%, rgba(0, 180, 255, 0.12), transparent 45%),
            linear-gradient(180deg, #03040a 0%, #070b1e 60%, #03040a 100%);
        color: #e6e6e6;
    }

    /* ===== SUBTLE NEON GRID ===== */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background:
            repeating-linear-gradient(
                90deg,
                rgba(0,255,255,0.03) 0px,
                rgba(0,255,255,0.03) 1px,
                transparent 1px,
                transparent 140px
            ),
            repeating-linear-gradient(
                0deg,
                rgba(255,0,255,0.025) 0px,
                rgba(255,0,255,0.025) 1px,
                transparent 1px,
                transparent 140px
            );
        pointer-events: none;
        z-index: -1;
    }

    /* ===== HEADINGS ===== */
    h1 {
        text-align: center;
        font-weight: 800;
        letter-spacing: 1px;
        text-shadow: 0 0 12px rgba(0,255,255,0.35);
    }

    h2, h3 {
        text-align: center;
        letter-spacing: 0.6px;
    }

    /* ===== TEXT AREA ===== */
    textarea {
        background-color: rgba(255,255,255,0.06) !important;
        color: #e6e6e6 !important;
        border-radius: 14px !important;
        border: 1px solid rgba(0,255,255,0.35) !important;
        box-shadow: inset 0 0 10px rgba(0,255,255,0.15);
    }

    /* ===== BUTTON ===== */
    .stButton > button {
        background: linear-gradient(90deg, #00e5ff, #8a2be2);
        color: black;
        font-weight: 800;
        border-radius: 14px;
        padding: 0.7rem 2rem;
        border: none;
        box-shadow: 0 0 22px rgba(0,229,255,0.6);
        transition: all 0.25s ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 0 32px rgba(138,43,226,0.85);
    }

    /* ===== RESULT BOX ===== */
    div[data-testid="stAlert"] {
        border-left: 4px solid #00e5ff;
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        box-shadow: 0 0 18px rgba(0,255,255,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1>🧠 Customer Review Sentiment Analyzer</h1>
    <p style="text-align:center; color:#9adfff; font-size:16px;">
        NLP-powered sentiment intelligence with a cyber-AI interface
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------- INPUT ----------------
review = st.text_area(
    "Customer Review",
    height=170,
    placeholder="Example: I really liked this product, the quality is excellent and delivery was fast.",
)

st.markdown(
    f"<p style='text-align:right; color:#7aa7c7;'>Characters: {len(review)}</p>",
    unsafe_allow_html=True,
)

# ---------------- ANALYZE BUTTON ----------------
if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a customer review.")
    else:
        # --------- 🔒 YOUR ML LOGIC (UNCHANGED) ---------
        processed = clean_text(review)
        vector = vectorizer.transform([processed])
        prediction = model.predict(vector)[0]

        decision = model.decision_function(vector)
        confidence_score = float(min(1.0, np.max(np.abs(decision)) / 3))
        # -----------------------------------------------

        st.subheader("Result")

        if prediction == "Positive":
            st.success(
                f"✅ **Positive Review**\n\n"
                f"📊 **Confidence:** {confidence_score:.2f}"
            )
        else:
            st.error(
                f"❌ **Negative Review**\n\n"
                f"📊 **Confidence:** {confidence_score:.2f}"
            )

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr style="border:1px solid rgba(0,255,255,0.15);">
    <p style="text-align:center; color:#6b8aa6; font-size:13px;">
        TF-IDF • SVM • NLP • Cyber-AI Interface
    </p>
    """,
    unsafe_allow_html=True,
)
