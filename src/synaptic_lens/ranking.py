"""Deterministic descriptive ranks and binary AUC (ties receive half credit)."""

import torch


def average_ranks(values):
    """Return zero-based average ranks on CPU in double precision."""
    values = values.double().flatten().cpu()
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def binary_auc(scores, positive):
    """Rank AUC with ties at half credit; unsupported outcomes remain null."""
    scores, positive = scores.flatten().cpu(), positive.flatten().bool().cpu()
    positives = int(positive.sum())
    negatives = len(scores) - positives
    if not positives or not negatives:
        return None
    ranks = average_ranks(scores)
    return float(
        (ranks[positive].sum() - positives * (positives - 1) / 2) / (positives * negatives)
    )
