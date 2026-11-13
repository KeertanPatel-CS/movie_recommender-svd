import numpy as np


def bias_baseline(R):
    """Overall mean plus one user effect and one movie effect."""
    known = ~np.isnan(R)
    mu = np.nanmean(R)

    b_user = np.nanmean(R - mu, axis=1)
    residual = R - mu - b_user[:, None]

    n_item = known.sum(axis=0)
    b_item = np.zeros(R.shape[1])
    has_rating = n_item > 0

    b_item[has_rating] = (
        np.nansum(residual, axis=0)[has_rating] / n_item[has_rating]
    )

    return mu, b_user, b_item

def baseline(R):
    """Use the Milestone 2 baseline with the Milestone 3 function name."""
    return bias_baseline(R)


def baseline_matrix(R):
    """Create baseline predictions for every user and movie."""

    # Calculate the overall mean and user/movie effects.
    mu, b_user, b_item = baseline(R)

    # Combine a column of user effects with a row of movie effects.
    return mu + b_user[:, None] + b_item[None, :]


def rank_k(U, s, Vt, k):
    """Rebuild a matrix using its first k SVD components."""

    # Keep the first k components and multiply the pieces together.
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]


def iterative_svd_predict(R, k, n_iter=15):
    """Predict ratings by repeatedly updating missing residuals."""

    # Identify observed training ratings.
    known = ~np.isnan(R)

    # Calculate baseline predictions from the supplied ratings.
    B = baseline_matrix(R)

    # Keep observed residuals and start missing residuals at zero.
    X = np.where(known, R - B, 0.0)

    for step in range(n_iter):
        # Find the current rank-k approximation.
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        L = rank_k(U, s, Vt, k)

        # Restore observed residuals and update only the gaps.
        X = np.where(known, R - B, L)

    # Return predictions on the original star-rating scale.
    return L + B
