"""Label-shift correction: re-estimate the team mix from the model's own
predictions, without labels, and reweight its probabilities.

The method is the EM algorithm of Saerens, Latinne and Decaestecker (2002),
"Adjusting the outputs of a classifier to new a priori probabilities". The
model was calibrated on validation, whose team mix is the source prior. If
the live mix differs (on the Jan-Feb 2026 holdout Credit Reporting fell from
61% to 2%), the posteriors are wrong by the ratio of the two priors. EM finds
the target prior that makes the adjusted posteriors self-consistent.

Offline: `estimate_prior` on a whole split. Online: `RollingPrior` keeps the
last N stacker outputs and re-estimates every `refresh` requests, so a change
in the mix of incoming complaints is picked up without retraining.
"""
from __future__ import annotations

from collections import deque

import numpy as np

MIN_PRIOR = 1e-4


def adjust(P: np.ndarray, source_prior: np.ndarray,
           target_prior: np.ndarray) -> np.ndarray:
    """Posteriors under `target_prior` for a model calibrated to `source_prior`."""
    P = np.asarray(P, dtype=np.float64)
    w = np.asarray(target_prior, dtype=np.float64) / np.asarray(source_prior,
                                                                dtype=np.float64)
    Q = P * w[None, :]
    s = Q.sum(axis=1, keepdims=True)
    return np.where(s > 0, Q / np.where(s > 0, s, 1.0), P)


def estimate_prior(P: np.ndarray, source_prior: np.ndarray,
                   max_iter: int = 200, tol: float = 1e-6,
                   min_prior: float = MIN_PRIOR) -> tuple[np.ndarray, int]:
    """EM estimate of the class prior behind the unlabelled rows in P.
    Returns (prior, iterations). A floor keeps every team reachable."""
    P = np.asarray(P, dtype=np.float64)
    source_prior = np.asarray(source_prior, dtype=np.float64)
    w = source_prior.copy()
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        Q = adjust(P, source_prior, w)
        w_new = np.maximum(Q.mean(axis=0), min_prior)
        w_new /= w_new.sum()
        done = np.abs(w_new - w).max() < tol
        w = w_new
        if done:
            break
    return w, n_iter


def prior_from_labels(labels, classes) -> np.ndarray:
    labels = np.asarray(labels).astype(str)
    counts = np.array([(labels == str(c)).sum() for c in classes],
                      dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    return counts / counts.sum()


class RollingPrior:
    """Online prior estimate over the last `window` probability vectors.

    Until `min_rows` vectors have been seen the source prior is used, so a
    freshly started service behaves exactly like the offline evaluation. The
    estimate is refreshed every `refresh` observations, not on every request,
    because EM over the window costs more than a prediction.
    """

    def __init__(self, source_prior: np.ndarray, window: int = 5000,
                 min_rows: int = 500, refresh: int = 100,
                 max_ratio: float = 50.0):
        self.source_prior = np.asarray(source_prior, dtype=np.float64)
        self.window = window
        self.min_rows = min_rows
        self.refresh = refresh
        self.max_ratio = max_ratio
        self._buf: deque = deque(maxlen=window)
        self._since_refresh = 0
        self.prior = self.source_prior.copy()
        self.n_estimates = 0

    @property
    def n_seen(self) -> int:
        return len(self._buf)

    @property
    def active(self) -> bool:
        return self.n_estimates > 0

    def observe(self, p: np.ndarray) -> None:
        self._buf.append(np.asarray(p, dtype=np.float64))
        self._since_refresh += 1
        if (len(self._buf) >= self.min_rows
                and self._since_refresh >= self.refresh):
            self._since_refresh = 0
            est, _ = estimate_prior(np.vstack(self._buf), self.source_prior)
            # Bound the correction so a short burst of one team cannot
            # swing the prior by orders of magnitude.
            lo = self.source_prior / self.max_ratio
            hi = np.minimum(self.source_prior * self.max_ratio, 1.0)
            est = np.clip(est, lo, hi)
            self.prior = est / est.sum()
            self.n_estimates += 1

    def adjust(self, p: np.ndarray) -> np.ndarray:
        if not self.active:
            return np.asarray(p, dtype=np.float64)
        return adjust(np.asarray(p)[None, :], self.source_prior, self.prior)[0]

    def reset(self) -> None:
        self._buf.clear()
        self._since_refresh = 0
        self.prior = self.source_prior.copy()
        self.n_estimates = 0
