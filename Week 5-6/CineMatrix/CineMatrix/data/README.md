# Dataset — MovieLens 100K

**Kaggle link:** https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset

## Download steps

### Option A — Kaggle website
1. Go to the link above and click "Download" (you'll need a free Kaggle account).
2. Unzip it. You'll get a folder (often called `ml-100k`) containing files like `u.data`, `u.item`, `u.user`.
3. Copy `u.data` and `u.item` into this `data/` folder.

### Option B — Kaggle API (faster, scriptable)
```bash
pip install kaggle
# place your kaggle.json API token in ~/.kaggle/kaggle.json first
kaggle datasets download -d prajitdatta/movielens-100k-dataset -p data/ --unzip
```

## Files you need

- **`u.data`** — the ratings file, tab-separated: `user_id  movie_id  rating  timestamp`
- **`u.item`** — movie metadata, pipe-separated: `movie_id | title | release_date | ... `
  (used to show movie titles in recommendations instead of just IDs)

## Alternative (bigger, if you want more of a challenge)

MovieLens 1M or 20M: https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset
The code in `src/data_loader.py` works the same way, just point it at the new file path —
you may want to subsample it since 20M ratings will make training slower.
