"""
FastAPI application for text content recommendations.
"""
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import joblib
from annoy import AnnoyIndex

import data_processing
from query import TextContentSearcher

# --- Configuration ---
CONFIG = {
    "tfidf_model_path": "models/tfidf_model.pkl",
    "svd_model_path": "models/svd_model.pkl",
    "annoy_index_path": "models/content_index.ann",
    "text_data_path": "data/sample_posts.csv",
}

# --- Dependency Injection: Data ---
def get_text_data():
    try:
        return data_processing.load_data(CONFIG["text_data_path"])
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Data file not found: {e}") from e

# --- Dependency Injection: Text Searcher ---
def get_text_searcher(data=Depends(get_text_data)):
    try:
        tfidf = joblib.load(CONFIG["tfidf_model_path"])
        svd = joblib.load(CONFIG["svd_model_path"])
        n_components = svd.n_components
        index = AnnoyIndex(n_components, 'angular')
        index.load(CONFIG["annoy_index_path"])
        return TextContentSearcher(tfidf, svd, index, data)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model file not found: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text Model Error: {e}") from e

# --- FastAPI Setup ---
app = FastAPI(title="STRV Text Content Recommender")

class TextQueryRequest(BaseModel):
    """
    Represents a request for text content recommendations.
    """
    text: str = Field(..., min_length=1, description="The query text.")
    top_k: int = Field(5, ge=1, le=100, description="Number of recommendations.")

@app.post("/recommend")
async def recommend_text(
    request: TextQueryRequest, searcher: TextContentSearcher = Depends(get_text_searcher)
):
    """Get similar text content recommendations."""
    try:
        results = searcher.find_similar(request.text, request.top_k)
        return {"query": request.text, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/health")
def health_check(text_searcher: TextContentSearcher = Depends(get_text_searcher)):
    """Service health monitor."""
    assert text_searcher is not None, "Text Searcher dependency failed"
    return {
        "status": "OK",
        "components": {
            "text_model": True,
            "database": False,
        },
    }
    