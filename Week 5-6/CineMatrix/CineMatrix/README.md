# 🎬 CineMatrix — A Movie Recommender Built From the Math Up

**Hack-o-Week (Weeks 5–6) project | Topics covered: vectors, matrices, dot products, eigenvalues, derivatives, gradients, chain rule**

## Why this project (and not another generic one)

Most "beginner ML" hackathon projects (digit classifier, house price predictor, sentiment analysis)
use libraries that hide the math. This project does the opposite: you will **derive and code the
gradient updates yourself**, using nothing but NumPy, and see *exactly* where vectors, matrices,
dot products, eigenvalues, derivatives and the chain rule show up in a real, working recommender
system — the same core idea Netflix used to win the $1M Netflix Prize.

## The idea in one line

Given a sparse **user × movie rating matrix R**, learn two smaller matrices **P** (users × k) and
**Q** (movies × k) such that `R ≈ P · Qᵀ`. Each user and each movie gets a "latent vector" in a
k-dimensional taste space, and predicting a rating is just a **dot product** of two vectors. We learn
P and Q with **gradient descent**, deriving the gradients ourselves using the **chain rule**.

## Where each topic actually appears

| Topic | Where it shows up in this project |
|---|---|
| Vectors | Each user/movie is represented as a latent-feature vector |
| Matrices | The rating matrix R, and factor matrices P, Q |
| Dot product | Predicted rating = `user_vector · movie_vector` |
| Eigenvalues (intuition) | Comparing our learned factorization to true SVD (`numpy.linalg.svd`); eigenvalues of `RᵀR` explain how much "taste variance" each latent dimension captures — used in `src/visualize.py` for 2D latent-space plots |
| Derivatives / Gradients | Loss = squared error + regularization; we derive `∂Loss/∂P` and `∂Loss/∂Q` by hand |
| Chain rule | Error term `e = r - p·q` is a composed function; gradient of the square-error w.r.t. `p` and `q` requires the chain rule (shown step-by-step in the notebook and in code comments) |

## Project structure

```
CineMatrix/
├── README.md                              <- you are here
├── requirements.txt
├── data/
│   └── README.md                          <- Kaggle dataset link + download instructions
├── src/
│   ├── data_loader.py                     <- loads ratings.csv -> sparse matrix
│   ├── matrix_factorization.py            <- the from-scratch MF model (gradients, chain rule)
│   ├── visualize.py                       <- loss curves, eigenvalue/PCA plots
│   └── recommend.py                       <- top-N recommendations for a user
├── notebook/
│   └── CineMatrix_MatrixFactorization.ipynb   <- full walkthrough with math markdowns
└── outputs/                                <- trained factors / plots get saved here
```

## Dataset

**MovieLens 100K** on Kaggle:
https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset

(~100,000 ratings from 943 users on 1,682 movies — small enough to train in minutes on a laptop,
big enough to be a real sparse-matrix problem.) See `data/README.md` for exact download steps.

## How to run

```bash
pip install -r requirements.txt
# download dataset into data/ (see data/README.md)
jupyter notebook notebook/CineMatrix_MatrixFactorization.ipynb
```

Or use the modules directly:

```python
from src.data_loader import load_ratings_matrix
from src.matrix_factorization import MatrixFactorization

R, user_map, movie_map = load_ratings_matrix("data/u.data")
mf = MatrixFactorization(n_factors=20, learning_rate=0.01, reg=0.02, n_epochs=50)
mf.fit(R)
```

## Stretch goals (for extra "not-so-generic" points during demo)

1. **Add a bias term** per user/movie (`r̂ = μ + b_u + b_i + p_u·q_i`) — derive the extra gradient terms.
2. **Compare** your hand-rolled gradient descent factorization against `numpy.linalg.svd` on the
   dense-filled matrix — show they converge to similar latent structure (this is where eigenvalues
   of `RᵀR` directly connect to singular values of R).
3. Build a **tiny Streamlit/Gradio UI**: pick a user, get live recommendations.
4. Plot the **loss surface** for a single (p, q) pair in 2D/3D to visually show what gradient
   descent is doing — great demo visual for judges.

## Other project ideas (if you want alternatives for the same topics)

- **PageRank from scratch** (eigenvalue problem: PageRank vector is the dominant eigenvector of the
  web-link matrix) applied to a citation network or Wikipedia link graph dataset.
- **Image compression via SVD** — show how truncating eigenvalues/singular values compresses an
  image, plot reconstruction error vs. number of components kept.
- **Optimizer visualizer** — implement gradient descent, momentum, and Adam from scratch on a 2D
  loss surface (e.g., Rosenbrock function) and animate the paths — pure calculus/chain-rule focus,
  no dataset needed.

CineMatrix is the recommended pick because it combines *both* linear algebra and calculus topics in
one coherent, demo-friendly, real-world system.
