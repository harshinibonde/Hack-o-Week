"""
visualize.py
-------------
Plots for the CineMatrix project:
  1. Training loss curve (gradient descent convergence)
  2. Eigenvalue-based 2D projection of learned movie vectors (connects the
     "eigenvalues" topic directly to the latent factors we trained)
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curve(loss_history, save_path=None):
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss (MSE + regularization)")
    plt.title("Gradient Descent Convergence")
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def eigen_project_2d(Q: np.ndarray):
    """
    Projects movie latent vectors Q (n_movies x k) onto their top-2
    eigenvectors of the covariance matrix QᵀQ (a classic PCA step).

    This is the direct link to "eigenvalues" in the syllabus:
    - Cov = QᵀQ / (n-1) is a (k x k) symmetric matrix
    - Its eigenvectors are the directions of maximum variance in taste-space
    - Its eigenvalues tell you HOW MUCH variance each direction explains
    """
    Q_centered = Q - Q.mean(axis=0)
    cov = (Q_centered.T @ Q_centered) / (Q_centered.shape[0] - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending order
    # Take the two eigenvectors with the largest eigenvalues
    order = np.argsort(eigenvalues)[::-1]
    top2 = eigenvectors[:, order[:2]]
    explained = eigenvalues[order[:2]] / eigenvalues.sum()

    projected = Q_centered @ top2  # (n_movies x 2)
    return projected, explained


def plot_movie_landscape(Q: np.ndarray, movie_ids, titles_map=None,
                          n_labels=15, save_path=None):
    """
    Scatter-plots movies in the 2D eigenvector space. Optionally labels a
    handful of movies so you can visually sanity-check that similar movies
    cluster together (e.g., all the action movies group up).
    """
    projected, explained = eigen_project_2d(Q)

    plt.figure(figsize=(8, 6))
    plt.scatter(projected[:, 0], projected[:, 1], alpha=0.5, s=15)
    plt.xlabel(f"Eigenvector 1 ({explained[0]*100:.1f}% variance explained)")
    plt.ylabel(f"Eigenvector 2 ({explained[1]*100:.1f}% variance explained)")
    plt.title("Learned Movie Taste-Space (projected onto top-2 eigenvectors)")
    plt.grid(alpha=0.3)

    if titles_map is not None:
        rng = np.random.default_rng(0)
        label_idx = rng.choice(len(movie_ids), size=min(n_labels, len(movie_ids)), replace=False)
        for idx in label_idx:
            mid = movie_ids[idx]
            title = titles_map.get(mid, str(mid))
            plt.annotate(title, (projected[idx, 0], projected[idx, 1]), fontsize=7, alpha=0.8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
