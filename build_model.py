from pathlib import Path
import sys
import joblib
from annoy import AnnoyIndex
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from data_processing import load_data
import os
from dotenv import load_dotenv

load_dotenv()

# --- Configuration (using environment variables) ---
MAX_FEATURES = int(os.getenv("MAX_FEATURES", 2000))
N_COMPONENTS = int(os.getenv("N_COMPONENTS", 50))
N_TREES = int(os.getenv("N_TREES", 20))
METRIC = os.getenv("METRIC", 'angular')
TEXT_DATA_PATH = os.getenv("TEXT_DATA_PATH", 'data/sample_posts.csv')
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))


def build_text_recommendation_system() -> None:
    """Builds the text-based recommendation model."""
    try:
        print("\n🔄 Loading and preprocessing text data...")
        df = load_data(TEXT_DATA_PATH)

        print("🔠 Creating TF-IDF vectors...")
        tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
        tfidf_vectors = tfidf.fit_transform(df['clean_text'])

        n_features = tfidf_vectors.shape[1]
        adjusted_components = min(N_COMPONENTS, n_features - 1)
        if adjusted_components != N_COMPONENTS:
            print(f"⚠️ Adjusted components from {N_COMPONENTS} to {adjusted_components}")

        print("📉 Reducing dimensions...")
        svd = TruncatedSVD(n_components=adjusted_components, random_state=42)
        reduced_vectors = svd.fit_transform(tfidf_vectors)

        print(f"🏗️ Building Annoy index ({adjusted_components} dimensions, {METRIC} metric)...")
        ann_index = AnnoyIndex(adjusted_components, METRIC)
        for idx, vec in enumerate(reduced_vectors):
            ann_index.add_item(idx, vec)
        ann_index.build(N_TREES)

        print("💾 Saving text models...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(tfidf, MODELS_DIR / "tfidf_model.pkl")
        joblib.dump(svd, MODELS_DIR / "svd_model.pkl")
        annoy_index_path = MODELS_DIR / "content_index.ann"
        ann_index.save(str(annoy_index_path.resolve()))

        print("✅ Text recommendation system built successfully!\n")

    except Exception as e:
        print(f"❌ Error building text model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build_text_recommendation_system()
    