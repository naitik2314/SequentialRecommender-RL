# src/item_feature_engineering.py

import os
import pandas as pd
import numpy as np

def build_item_features(raw_path: str, processed_path: str):
    """
    Reads genome-scores.csv and genome-tags.csv from raw,
    pivots into an [num_items × num_tags] relevance matrix,
    then saves as item_features.npy in processed_path.
    """
    genome_scores_path = os.path.join(raw_path, 'genome-scores.csv')
    genome_tags_path   = os.path.join(raw_path, 'genome-tags.csv')
    
    # Load data
    scores = pd.read_csv(genome_scores_path)
    tags   = pd.read_csv(genome_tags_path)
    
    # Pivot scores: rows=item_id, cols=tag_id
    item_tag_matrix = scores.pivot(
        index='item_id',
        columns='tag_id',
        values='relevance'
    ).fillna(0)
    
    # Optional: sort columns by tag name for interpretability
    tag_order = tags.set_index('tag_id').loc[item_tag_matrix.columns]['tag_name']
    item_tag_matrix.columns = tag_order
    
    # Convert to numpy array and save
    features = item_tag_matrix.values  # shape: [num_items, num_tags]
    os.makedirs(processed_path, exist_ok=True)
    np.save(os.path.join(processed_path, 'item_features.npy'), features)
    
    print(f"[feature_engineering] Saved item_features.npy with shape {features.shape}")

if __name__ == '__main__':
    RAW_PATH       = os.path.join('data', 'raw', 'ml-25m')
    PROCESSED_PATH = os.path.join('data', 'processed')
    build_item_features(RAW_PATH, PROCESSED_PATH)
