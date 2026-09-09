"""
matrix_factorization.py
------------------------
From-scratch Matrix Factorization via Stochastic Gradient Descent.
No sklearn/torch/tensorflow used for the core algorithm -- just NumPy --
so every line maps directly to the math.

----------------------------------------------------------------------
THE MATH (this is the part your Hack-o-Week write-up / demo should lean on)
----------------------------------------------------------------------

We want to approximate the rating matrix R (n_users x n_movies) as:

    R  ≈  P · Qᵀ

where:
    P is (n_users  x k)   -> each row p_u is user u's latent taste VECTOR
    Q is (n_movies x k)   -> each row q_i is movie i's latent taste VECTOR
    k is the number of latent factors (a hyperparameter, e.g. 20)

Predicted rating for user u, movie i is a DOT PRODUCT of two vectors:

    r_hat(u, i) = p_u · q_i = sum_f( P[u, f] * Q[i, f] )

LOSS (only over observed ratings, i.e. where R[u, i] != 0), with L2 regularization:

    e(u, i) = r(u, i) - r_hat(u, i)                      <- error for one rating

    L = sum_over_observed( e(u,i)^2 )  +  reg * ( ||p_u||^2 + ||q_i||^2 )

GRADIENTS (this is where the CHAIN RULE comes in):

    e(u,i) = r(u,i) - p_u · q_i     is a function of p_u and q_i.
    L_ui   = e(u,i)^2 + reg*(||p_u||^2 + ||q_i||^2)   is a function of e(u,i).

    dL_ui/dp_u  = dL_ui/de * de/dp_u
                = 2*e(u,i) * (-q_i)   + 2*reg*p_u        <- chain rule!
                = -2*e(u,i)*q_i + 2*reg*p_u

    dL_ui/dq_i  = dL_ui/de * de/dq_i
                = 2*e(u,i) * (-p_u)   + 2*reg*q_i
                = -2*e(u,i)*p_u + 2*reg*q_i

We drop the constant 2 into the learning rate and get the classic SGD update rule
(used below in `fit`):

    p_u <- p_u + lr * ( e(u,i) * q_i  -  reg * p_u )
    q_i <- q_i + lr * ( e(u,i) * p_u  -  reg * q_i )

Each of these updates moves p_u and q_i a small step DOWNHILL on the loss
surface -- literally gradient descent, one observed rating at a time.
"""

import numpy as np


class MatrixFactorization:
    def __init__(self, n_factors: int = 20, learning_rate: float = 0.01,
                 reg: float = 0.02, n_epochs: int = 50, seed: int = 42,
                 verbose: bool = True):
        self.k = n_factors
        self.lr = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.seed = seed
        self.verbose = verbose

        self.P = None          # user latent matrix   (n_users  x k)
        self.Q = None          # movie latent matrix   (n_movies x k)
        self.loss_history = []  # for plotting convergence

    def fit(self, R: np.ndarray):
        rng = np.random.default_rng(self.seed)
        n_users, n_movies = R.shape

        # Initialize small random vectors -- if we started at all-zeros,
        # every gradient would be zero and nothing would ever learn.
        self.P = rng.normal(scale=0.1, size=(n_users, self.k))
        self.Q = rng.normal(scale=0.1, size=(n_movies, self.k))

        # Precompute list of observed (u, i, rating) triples once.
        users, items = np.nonzero(R)
        ratings = R[users, items]
        observed = list(zip(users, items, ratings))

        for epoch in range(1, self.n_epochs + 1):
            rng.shuffle(observed)
            for u, i, r in observed:
                pu = self.P[u]
                qi = self.Q[i]

                pred = np.dot(pu, qi)     # r_hat(u,i) -- the dot product
                err = r - pred            # e(u,i)

                # Gradient-descent updates derived above via the chain rule
                self.P[u] = pu + self.lr * (err * qi - self.reg * pu)
                self.Q[i] = qi + self.lr * (err * pu - self.reg * qi)

            epoch_loss = self._compute_loss(observed)
            self.loss_history.append(epoch_loss)
            if self.verbose and (epoch == 1 or epoch % 5 == 0):
                print(f"Epoch {epoch:3d}/{self.n_epochs} | loss = {epoch_loss:.4f}")

        return self

    def _compute_loss(self, observed):
        total = 0.0
        for u, i, r in observed:
            pred = np.dot(self.P[u], self.Q[i])
            err = r - pred
            total += err ** 2
            total += self.reg * (np.dot(self.P[u], self.P[u]) + np.dot(self.Q[i], self.Q[i]))
        return total / len(observed)

    def predict(self, u: int, i: int) -> float:
        """Predicted rating = dot product of the two latent vectors."""
        return float(np.dot(self.P[u], self.Q[i]))

    def full_prediction_matrix(self) -> np.ndarray:
        """R_hat = P . Q^T -- the full reconstructed rating matrix."""
        return self.P @ self.Q.T

    def rmse(self, triples) -> float:
        """Root-mean-squared-error over a list of (u, i, rating) triples."""
        sq_errors = []
        for u, i, r in triples:
            pred = self.predict(u, i)
            sq_errors.append((r - pred) ** 2)
        return float(np.sqrt(np.mean(sq_errors)))
