import pandas as pd
import os

def preprocess_movielens_25m(raw_path: str, processed_path: str):
    """
    Parse and clean the MovieLens 25M dataset.
    Expects raw_path to contain 'ratings.csv', 'movies.csv', and 'links.csv'
    Outputs:
      - ratings_clean.csv
      - movies.csv
      - movies_enriched.csv (if links.csv exists)
    """

    # Ensure output directory exists
    os.makedirs(processed_path, exist_ok=True)

    # Loading RAW CSVs
    ratings_in  = os.path.join(raw_path, 'ratings.csv')
    movies_in   = os.path.join(raw_path, 'movies.csv')
    links_in    = os.path.join(raw_path, 'links.csv')

    ratings = pd.read_csv(ratings_in)
    movies  = pd.read_csv(movies_in)

    # Handling missing values
    ratings_clean = ratings.dropna().copy()
    ratings_clean.to_csv(os.path.join(processed_path, 'ratings_clean.csv'), index=False)
    print(f"[preprocess] ratings_clean.csv: {len(ratings)} → {len(ratings_clean)} rows")

    # Save the raw movies metadata
    movies.to_csv(os.path.join(processed_path, 'movies.csv'), index=False)
    print(f"[preprocess] movies.csv saved ({len(movies)} rows)")

    # Enrich movies metadata if links.csv is present
    if os.path.exists(links_in):
        links = pd.read_csv(links_in)
        # ml-25m links columns: movieId, imdbId, tmdbId
        enriched = movies.merge(links, on='movieId', how='left')
        enriched.to_csv(os.path.join(processed_path, 'movies_enriched.csv'), index=False)
        print(f"[preprocess] movies_enriched.csv saved ({len(enriched)} rows)")
    else:
        print("[preprocess] links.csv not found; skipping enrichment")

if __name__ == '__main__':
    RAW_PATH      = os.path.join('data', 'raw', 'ml-25m')
    PROCESSED_PATH = os.path.join('data', 'processed')

    if not os.path.isdir(RAW_PATH):
        raise FileNotFoundError(f"Expected raw data at {RAW_PATH}")

    preprocess_movielens_25m(RAW_PATH, PROCESSED_PATH)
    print("MovieLens 25M preprocessing complete.")