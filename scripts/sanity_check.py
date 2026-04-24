# sanity_check.py
# ------------------------------------------------------------
# Validates full pipeline:
# - Model loading
# - Feature extraction
# - Prediction correctness
# - Quick accuracy check
# ------------------------------------------------------------

import pathlib
import pickle
import sys
import random

# ============================
# 📁 PATH SETUP
# ============================

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features import extract_features  # main pipeline

MODEL_PATH = ROOT / "models" / "voice_model.pkl"
REAL_DIR = ROOT / "data" / "real"
FAKE_DIR = ROOT / "data" / "fake"

SUPPORTED = {".wav"}

LABEL_MAP = {0: "REAL", 1: "FAKE"}

# Number of test samples per class
SAMPLES_PER_CLASS = 3


# ============================
# 🤖 LOAD MODEL
# ============================

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"❌ Model not found at {MODEL_PATH}\nRun train.py first."
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print(f"[ok] Model loaded: {type(model).__name__}")
    return model


# ============================
# 📂 GET RANDOM FILES
# ============================

def get_random_files(directory, n=3):
    files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED
    ]

    if not files:
        return []

    return random.sample(files, min(n, len(files)))


# ============================
# 🔍 SINGLE INFERENCE
# ============================

def run_inference(model, filepath, expected):
    try:
        features = extract_features(str(filepath))

        pred = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0]

        pred_label = LABEL_MAP[pred]
        confidence = proba[pred] * 100

        match = (pred_label == expected)

        icon = "✅" if match else "❌"

        print(f"\nFile: {filepath.name}")
        print(f"Expected: {expected}")
        print(f"Predicted: {pred_label} ({confidence:.1f}%) {icon}")
        print(f"REAL={proba[0]*100:.1f}% | FAKE={proba[1]*100:.1f}%")

        return match

    except Exception as e:
        print(f"[error] {filepath.name} → {e}")
        return False


# ============================
# 🚀 MAIN
# ============================

def main():
    print("=" * 60)
    print("🧠 Orizion — Sanity Check v2")
    print("=" * 60)

    # Load model
    model = load_model()

    # Get samples
    real_files = get_random_files(REAL_DIR, SAMPLES_PER_CLASS)
    fake_files = get_random_files(FAKE_DIR, SAMPLES_PER_CLASS)

    if not real_files and not fake_files:
        print("❌ No audio files found. Add data and retry.")
        return

    print("\n🔍 Running tests...")
    print("-" * 60)

    results = []

    # Test REAL
    for file in real_files:
        results.append(run_inference(model, file, "REAL"))

    # Test FAKE
    for file in fake_files:
        results.append(run_inference(model, file, "FAKE"))

    # ============================
    # 📊 SUMMARY
    # ============================

    total = len(results)
    passed = sum(results)

    print("\n" + "=" * 60)
    print(f"📊 Accuracy Snapshot: {passed}/{total} correct")

    accuracy = (passed / total) * 100 if total > 0 else 0

    print(f"📈 Accuracy: {accuracy:.1f}%")

    if accuracy > 80:
        print("Status: ✅ STRONG")
    elif accuracy > 60:
        print("Status: ⚠️ MODERATE")
    else:
        print("Status: ❌ WEAK — check pipeline")

    print("=" * 60)


# ============================
# ▶ RUN
# ============================

if __name__ == "__main__":
    main()