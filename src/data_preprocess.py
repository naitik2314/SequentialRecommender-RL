import pandas as pd
import os

def preprocess_movielens_25m(raw_path: str, processed_path: str):
    """
    Parse and clean the MovieLens 25M dataset.
    Expects raw_path to contain 'ratings.csv', 'movies.csv', and 'links.csv'
    """