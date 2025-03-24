import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from pathlib import Path

nltk.download(['stopwords', 'wordnet'], quiet=True)

def clean_text(text):
    """Clean and lemmatize text input."""
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    return ' '.join([
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ])

def load_data(data_path):
    """Load and preprocess CSV data."""
    df = pd.read_csv(data_path)
    df['clean_text'] = df['text'].apply(clean_text)
    return df

def main():
    """Main execution (for testing)"""
    df = load_data('data/sample_posts.csv')  # Ensure this path is correct
    print("\n✅ Data processing successful!")
    print("\nFirst 5 entries:")
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    main()
    