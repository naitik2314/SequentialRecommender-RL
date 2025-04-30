# src/item_feature_engineering.py

import os
import pandas as pd
import numpy as np

def build_item_features(raw_path: str, processed_path: str):
    """
    Reads genome-scores.csv and genome-tags.csv from raw_path,
    pivots into an [num_items × num_tags] relevance matrix (indexed by movieId),
    renames columns to human-readable tag names, and saves as item_features.npy.
    """
    # Input files
    scores_path = os.path.join(raw_path, 'genome-scores.csv')
    tags_path   = os.path.join(raw_path, 'genome-tags.csv')

    # Load
    scores = pd.read_csv(scores_path)      # columns: movieId, tagId, relevance
    tags   = pd.read_csv(tags_path)        # columns: tagId, tagName

    # Pivot into item×tag matrix
    item_tag = scores.pivot(
        index='movieId',
        columns='tagId',
        values='relevance'
    ).fillna(0)

    # Map tagId → tagName for column labels
    tag_map = tags.set_index('tagId')['tagName']
    item_tag.columns = [tag_map[tag] for tag in item_tag.columns]

    # Convert to numpy and save
    features = item_tag.values  # shape: [num_movies, num_tags]
    os.makedirs(processed_path, exist_ok=True)
    np.save(os.path.join(processed_path, 'item_features.npy'), features)

    print(f"[feature_engineering] Saved item_features.npy with shape {features.shape}")

if __name__ == '__main__':
    RAW_PATH       = os.path.join('data', 'raw', 'ml-25m')
    PROCESSED_PATH = os.path.join('data', 'processed')
    build_item_features(RAW_PATH, PROCESSED_PATH)
