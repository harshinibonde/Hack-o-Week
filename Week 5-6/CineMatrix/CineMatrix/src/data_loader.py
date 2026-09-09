"""
data_loader.py
----------------
Loads the MovieLens ratings file and builds the sparse-ish user x movie
rating matrix R used everywhere else in this project.

R is a matrix (vectors stacked as rows/columns) where R[u, i] = rating
given by user u to movie i, and 0 where no rating exists (unobserved).
"""

import numpy as np
import pandas as pd


def load_ratings_matrix(ratings_path: str):
    """
    Load u.data (tab-separated: user_id, movie_id, rating, timestamp)
    and return:
        R          : np.ndarray of shape (n_users, n_movies), 0 = missing
        user_map   : dict {original_user_id -> row index in R}
        movie_map  : dict {original_movie_id -> column index in R}
    """
    df = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
    )

    unique_users = sorted(df["user_id"].unique())
    unique_movies = sorted(df["movie_id"].unique())

    user_map = {uid: idx for idx, uid in enumerate(unique_users)}
    movie_map = {mid: idx for idx, mid in enumerate(unique_movies)}

    n_users, n_movies = len(unique_users), len(unique_movies)
    R = np.zeros((n_users, n_movies), dtype=np.float32)

    for row in df.itertuples(index=False):
        u = user_map[row.user_id]
        i = movie_map[row.movie_id]
        R[u, i] = row.rating

    return R, user_map, movie_map


def load_movie_titles(items_path: str):
    """
    Load u.item (pipe-separated). Returns dict {movie_id -> title}.
    u.item has no header; columns are:
    movie_id | title | release_date | video_release_date | IMDb_URL | 19 genre flags...
    """
    titles = {}
    with open(items_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            movie_id = int(parts[0])
            title = parts[1]
            titles[movie_id] = title
    return titles


def train_test_split_matrix(R: np.ndarray, test_ratio: float = 0.2, seed: int = 42):
    """
    Splits observed (nonzero) entries of R into a train matrix and a held-out
    set of (u, i, rating) triples for evaluation. This keeps R's shape intact
    (needed since P, Q dimensions depend on n_users, n_movies) but hides some
    ratings from training so we can measure RMSE on unseen data.
    """
    rng = np.random.default_rng(seed)
    users, items = np.nonzero(R)
    n = len(users)
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)
    test_idx = idx[:n_test]

    R_train = R.copy()
    test_triples = []
    for k in test_idx:
        u, i = users[k], items[k]
        test_triples.append((u, i, R[u, i]))
        R_train[u, i] = 0.0

    return R_train, test_triples
