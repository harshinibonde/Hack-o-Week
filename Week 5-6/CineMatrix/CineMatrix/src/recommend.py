"""
recommend.py
-------------
Given a trained MatrixFactorization model, generate top-N movie
recommendations for a given user by ranking predicted ratings
(dot products of the user's vector with every movie's vector).
"""

import numpy as np


def recommend_for_user(mf, user_row: int, R_train: np.ndarray,
                        movie_map: dict, titles_map: dict = None, top_n: int = 10):
    """
    mf         : trained MatrixFactorization instance
    user_row   : row index of the user in R (use user_map[original_id] to get this)
    R_train    : the matrix used for training (so we can exclude already-rated movies)
    movie_map  : dict {original_movie_id -> column index}, used to map back to real IDs
    titles_map : optional dict {movie_id -> title} from load_movie_titles()
    """
    inv_movie_map = {v: k for k, v in movie_map.items()}

    # Predicted rating for every movie = dot product of user vector with
    # every movie vector -- this is literally P[u] @ Q.T, one big
    # matrix-vector multiplication built from many dot products.
    scores = mf.P[user_row] @ mf.Q.T  # shape (n_movies,)

    already_rated = np.nonzero(R_train[user_row])[0]
    scores[already_rated] = -np.inf  # don't recommend what they've already rated

    top_indices = np.argsort(scores)[::-1][:top_n]

    recommendations = []
    for idx in top_indices:
        movie_id = inv_movie_map[idx]
        title = titles_map.get(movie_id, f"Movie #{movie_id}") if titles_map else f"Movie #{movie_id}"
        recommendations.append((title, float(scores[idx])))

    return recommendations
