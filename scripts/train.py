# train.py
# ------------------------------------------------------------
# Builds dataset, trains classifier, evaluates, and saves model
# Supports flexible feature pipelines (MFCC / Wav2Vec ready)
# ------------------------------------------------------------

import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ============================
# 📁 PATH SETUP
# ============================

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features import extract_features  # change here if switching later

DATA_DIRS = {
    ROOT / "data" / "real": 0,
    ROOT / "data" / "fake": 1,
}

OUTPUT_CSV = ROOT / "outputs" / "dataset.csv"
MODEL_PATH = ROOT / "models" / "voice_model.pkl"

SUPPORTED_EXTS = {".wav"}

# ============================
# ⚙️ CONFIG
# ============================

TEST_SIZE = 0.2
RANDOM_STATE = 42

USE_SVM = False   # 🔁 toggle model
USE_SCALING = True

# ============================
# 📊 DATASET BUILDING
# ============================

def build_dataset():
    X, y = [], []

    print("\n🔍 Extracting features...")

    for directory, label in DATA_DIRS.items():
        if not directory.exists():
            print(f"[skip] {directory}")
            continue

        files = [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        ]

        for file in files:
            try:
                features = extract_features(str(file)).flatten()

                if np.any(np.isnan(features)):
                    print(f"[skip] NaN in {file.name}")
                    continue

                X.append(features)
                y.append(label)

                tag = "REAL" if label == 0 else "FAKE"
                print(f"[ok] [{tag}] {file.name}")

            except Exception as e:
                print(f"[error] {file.name} → {e}")

    if not X:
        raise RuntimeError("No valid features extracted!")

    X = np.array(X)
    y = np.array(y)

    print(f"\n📊 Dataset shape: {X.shape}")

    return X, y


# ============================
# 💾 SAVE DATASET
# ============================

def save_dataset(X, y):
    df = pd.DataFrame(X)
    df["label"] = y

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"[saved] Dataset → {OUTPUT_CSV}")


# ============================
# 🤖 MODEL TRAINING
# ============================

def build_model():
    if USE_SVM:
        model = SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE)
        print("⚙️ Using SVM")
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        print("⚙️ Using Random Forest")

    if USE_SCALING:
        print("⚙️ Scaling enabled")
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

    return model


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n" + "=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("=" * 60)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

    return model


# ============================
# 💾 SAVE MODEL
# ============================

def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"[saved] Model → {MODEL_PATH}")


# ============================
# 🚀 MAIN
# ============================

def main():
    print("=" * 60)
    print("🧠 Orizion — Training Pipeline v2")
    print("=" * 60)

    X, y = build_dataset()
    save_dataset(X, y)

    model = train_model(X, y)
    save_model(model)

    print("\n✅ Training complete\n")


if __name__ == "__main__":
    main()