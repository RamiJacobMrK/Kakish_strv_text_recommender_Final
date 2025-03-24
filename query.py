import joblib
from annoy import AnnoyIndex
from data_processing import clean_text

class TextContentSearcher:
    """Recommendation engine for finding similar text content."""
    def __init__(self, tfidf, svd, index, df):
        """Initializes TextContentSearcher with pre-loaded models and data."""
        self.tfidf = tfidf
        self.svd = svd
        self.index = index
        self.df = df

    def find_similar(self, query: str, top_k: int = 5) -> list:
        """Find content similar to input query."""
        clean_query = clean_text(query)
        tfidf_vec = self.tfidf.transform([clean_query])
        reduced_vec = self.svd.transform(tfidf_vec)
        indices = self.index.get_nns_by_vector(reduced_vec[0], top_k)
        return self.df.iloc[indices].to_dict(orient='records')
    