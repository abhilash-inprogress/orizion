# preprocess.py
# ------------------------------------------------------------
# Standardizes all audio files in data/real and data/fake:
# - Converts to mono
# - Resamples to 16 kHz
# - Saves as 16-bit PCM WAV
# - Optionally removes original non-wav files
# ------------------------------------------------------------

import pathlib
import librosa
import soundfile as sf

# ============================
# ⚙️ CONFIGURATION
# ============================

SAMPLE_RATE = 16000  # target sample rate

SUPPORTED_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".mpeg"}

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]

DATA_DIRS = [
    BASE_DIR / "data" / "real",
    BASE_DIR / "data" / "fake",
]

# Toggle this if you want to KEEP originals
DELETE_ORIGINAL = True

# Debug/Stats tracking
STATS = {
    "total": 0,
    "processed": 0,
    "skipped": 0,
    "errors": 0,
    "by_ext": {}
}


# ============================
# 🎧 CORE FUNCTION
# ============================

def standardize_file(filepath):
    try:
        # Load audio (resample + mono)
        audio, _ = librosa.load(str(filepath), sr=SAMPLE_RATE, mono=True)

        # Always save as .wav with SAME NAME (no duplication)
        out_path = filepath.with_suffix(".wav")

        # Write file (this overwrites safely)
        sf.write(str(out_path), audio, SAMPLE_RATE, subtype="PCM_16")

        # Delete original ONLY if it was not wav
        if filepath.suffix.lower() != ".wav" and DELETE_ORIGINAL:
            filepath.unlink()

        STATS["processed"] += 1
        ext = filepath.suffix.lower()
        STATS["by_ext"][ext] = STATS["by_ext"].get(ext, 0) + 1
        
        print(f"[✓ processed] {filepath.name:30} → {out_path.name}")

    except Exception as e:
        STATS["errors"] += 1
        print(f"[✗ error   ] {filepath.name:30} → {str(e)[:50]}")


# ============================
# 📁 DIRECTORY PROCESSING
# ============================

def preprocess_directory(directory: pathlib.Path) -> None:
    """
    Process all valid audio files in a given directory.
    """

    if not directory.exists():
        print(f"\n[⚠️  skip] Directory not found: {directory}")
        return

    files = [
        f for f in sorted(directory.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    ]

    if not files:
        print(f"\n[⚠️  skip] No valid audio files in: {directory}")
        return

    print("\n" + "=" * 70)
    print(f"📁 {directory.name.upper()}")
    print(f"   Files to process: {len(files)}")
    print("=" * 70)

    for file in files:
        STATS["total"] += 1
        standardize_file(file)


# ============================
# 🚀 MAIN ENTRY
# ============================

def main() -> None:
    print("\n" + "=" * 70)
    print("🎧 Orizion — Audio Preprocessor")
    print(f"   Target: {SAMPLE_RATE} Hz | Mono | PCM-16 WAV")
    print(f"   Supported formats: {', '.join(sorted(SUPPORTED_EXTS))}")
    print("=" * 70)

    for directory in DATA_DIRS:
        preprocess_directory(directory)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("📊 PREPROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total files processed: {STATS['processed']}")
    print(f"Total errors:          {STATS['errors']}")
    print()
    print("By original format:")
    for ext in sorted(STATS["by_ext"].keys()):
        count = STATS["by_ext"][ext]
        print(f"  {ext:10} → {count:3} files converted to WAV")
    
    if STATS["processed"] > 0:
        print(f"\n✅ Preprocessing complete. {STATS['processed']} files standardized to 16kHz mono WAV.\n")
    else:
        print(f"\n⚠️  No files were processed.\n")


# ============================
# ▶ RUN
# ============================

if __name__ == "__main__":
    main()