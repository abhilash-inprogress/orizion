# app.py

import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import pickle
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import time

from features import extract_features

# ============================
# 🎨 PAGE CONFIG
# ============================

st.set_page_config(page_title="Orizion", layout="wide")

st.title("🎧 Orizion — Voice Authenticity Analyzer")
st.caption("AI vs Human Voice Detection System")

# ============================
# 🧠 MODEL LOAD
# ============================

@st.cache_resource
def load_model():
    try:
        with open("models/voice_model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

model = load_model()

# ============================
# 🎤 INPUT
# ============================

st.subheader("🎤 Input")

col1, col2 = st.columns(2)

audio_path = None

with col1:
    duration = st.slider("Recording Duration (sec)", 3, 15, 5)

    if st.button("🎤 Record Audio"):
        fs = 16000
        st.info("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(temp.name, fs, recording)

        audio_path = temp.name
        st.audio(temp.name)

with col2:
    uploaded = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg", "m4a"])

    if uploaded:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp.write(uploaded.read())
        temp.flush()

        audio_path = temp.name
        st.audio(temp.name)

# ============================
# ⚙️ PROCESSING
# ============================

if audio_path:

    st.divider()
    st.subheader("⚙️ Processing")

    steps = [
        "Loading audio",
        "Running VAD",
        "Extracting features",
        "Running model"
    ]

    for step in steps:
        st.write(f"✔ {step}")
        time.sleep(0.4)

    y, sr = librosa.load(audio_path, sr=16000)

    # ============================
    # 🔊 SIMPLE VAD
    # ============================

    energy = np.abs(y)
    threshold = np.mean(energy) * 0.5

    speech = energy > threshold
    speech_ratio = np.sum(speech) / len(speech)

    duration_sec = len(y) / sr

    # ============================
    # 📊 SUMMARY
    # ============================

    st.divider()
    st.subheader("📊 Audio Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Duration", f"{duration_sec:.2f}s")
    col2.metric("Speech Ratio", f"{speech_ratio*100:.1f}%")
    col3.metric("Sample Rate", f"{sr} Hz")

    # ============================
    # 🤖 RESULT
    # ============================

    st.divider()
    st.subheader("🤖 Result")

    features = extract_features(audio_path)

    if model:
        pred = model.predict(features)[0]
        probs = model.predict_proba(features)[0]

        confidence = max(probs) * 100

        st.metric("Confidence", f"{confidence:.1f}%")
        st.progress(confidence / 100)

        # confidence meaning
        if confidence > 80:
            st.success("High confidence prediction")
        elif confidence > 60:
            st.warning("Moderate confidence")
        else:
            st.error("Low confidence — uncertain result")

        if pred == 1:
            st.error("🚨 AI Voice Detected")
        else:
            st.success("✅ Human Voice")

    else:
        st.warning("Model not loaded")

    # ============================
    # 📈 VISUALS
    # ============================

    st.divider()
    st.subheader("📈 Audio Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Waveform")
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("Spectrogram")
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)))
        fig, ax = plt.subplots()
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        fig.colorbar(img)
        st.pyplot(fig)

    # ============================
    # 🔍 EXPLANATION
    # ============================

    st.divider()
    st.subheader("🔍 Explanation")

    with st.expander("Waveform"):
        st.write("""
Waveform shows amplitude over time.
Human speech contains irregular fluctuations,
while AI voices often appear smoother.
""")

    with st.expander("Spectrogram"):
        st.write("""
Spectrogram shows frequency vs time.
Human speech is chaotic and variable.
AI voices tend to show consistent patterns.
""")

    # ============================
    # 🧠 INTERPRETATION
    # ============================

    st.divider()
    st.subheader("🧠 Interpretation")

    st.write("""
The model analyzes acoustic patterns such as pitch variation,
energy distribution, and frequency dynamics.

Human voices exhibit natural imperfections and variability,
while AI-generated voices tend to be smoother and more consistent.
""")

    # ============================
    # ⚙️ TECHNICAL
    # ============================

    with st.expander("Technical Details"):
        st.write(f"Feature shape: {features.shape}")
        st.write(f"Prediction probabilities: {probs}")

    # ============================
    # ⚠️ DISCLAIMER
    # ============================

    st.caption("⚠️ This system is probabilistic. Results may vary based on audio quality.")