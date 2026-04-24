# features.py
# ------------------------------------------------------------
# Hybrid Feature Extraction:
# - Handcrafted acoustic features (primary)
# - Optional Wav2Vec embeddings (advanced)
# ------------------------------------------------------------

import numpy as np
import librosa

# ============================
# ⚙️ CONFIG
# ============================

SAMPLE_RATE = 16000
N_MFCC = 40
N_CHROMA = 12
N_CONTRAST = 6

USE_WAV2VEC = False  # 🔁 toggle advanced mode


# ============================
# 📛 FEATURE NAMES
# ============================

def feature_names():
    names = []

    names += [f"mfcc_mean_{i}" for i in range(N_MFCC)]
    names += [f"mfcc_delta_mean_{i}" for i in range(N_MFCC)]
    names += [f"mfcc_delta2_mean_{i}" for i in range(N_MFCC)]

    names += [f"chroma_mean_{i}" for i in range(N_CHROMA)]
    names += [f"chroma_std_{i}" for i in range(N_CHROMA)]

    names += [f"contrast_mean_{i}" for i in range(N_CONTRAST + 1)]
    names += [f"contrast_std_{i}" for i in range(N_CONTRAST + 1)]

    names += [f"tonnetz_mean_{i}" for i in range(6)]
    names += [f"tonnetz_std_{i}" for i in range(6)]

    names += [
        "zcr_mean", "zcr_std",
        "centroid_mean", "centroid_std",
        "bandwidth_mean", "bandwidth_std",
        "rolloff_mean", "rolloff_std",
        "rmse_mean", "rmse_std",
    ]

    return names


# ============================
# 🔧 UTILITY
# ============================

def mean_std(feature):
    return np.mean(feature, axis=1), np.std(feature, axis=1)


# ============================
# 🎧 HANDCRAFTED FEATURES
# ============================

def extract_handcrafted(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    chroma = librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE, n_chroma=N_CHROMA)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=SAMPLE_RATE, n_bands=N_CONTRAST)

    harmonic = librosa.effects.harmonic(audio)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=SAMPLE_RATE)

    zcr = librosa.feature.zero_crossing_rate(audio)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE)
    rmse = librosa.feature.rms(y=audio)

    # Mean + std
    mfcc_mean, _ = mean_std(mfcc)
    mfcc_delta_mean, _ = mean_std(mfcc_delta)
    mfcc_delta2_mean, _ = mean_std(mfcc_delta2)

    chroma_mean, chroma_std = mean_std(chroma)
    contrast_mean, contrast_std = mean_std(contrast)
    tonnetz_mean, tonnetz_std = mean_std(tonnetz)

    zcr_mean, zcr_std = mean_std(zcr)
    centroid_mean, centroid_std = mean_std(centroid)
    bandwidth_mean, bandwidth_std = mean_std(bandwidth)
    rolloff_mean, rolloff_std = mean_std(rolloff)
    rmse_mean, rmse_std = mean_std(rmse)

    features = np.concatenate([
        mfcc_mean,
        mfcc_delta_mean,
        mfcc_delta2_mean,
        chroma_mean,
        chroma_std,
        contrast_mean,
        contrast_std,
        tonnetz_mean,
        tonnetz_std,
        zcr_mean,
        zcr_std,
        centroid_mean,
        centroid_std,
        bandwidth_mean,
        bandwidth_std,
        rolloff_mean,
        rolloff_std,
        rmse_mean,
        rmse_std,
    ])

    return features


# ============================
# 🤖 WAV2VEC FEATURES (OPTIONAL)
# ============================

def extract_wav2vec(audio):
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    import torch

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    model.eval()

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).numpy().flatten()

    return embedding


# ============================
# 🎯 MAIN FUNCTION
# ============================

def extract_features(file_path: str) -> np.ndarray:
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

        # Edge case: silence
        if np.max(np.abs(audio)) < 1e-5:
            raise ValueError("Audio is silent or too weak")

        # Normalize
        audio = librosa.util.normalize(audio)

        # Base features
        handcrafted = extract_handcrafted(audio)

        if USE_WAV2VEC:
            wav2vec = extract_wav2vec(audio)
            combined = np.concatenate([handcrafted, wav2vec])
            return combined.reshape(1, -1)

        return handcrafted.reshape(1, -1)

    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {e}")