import numpy as np
# These implementations of the soft-label and perspectivist metrics 
# are adapted from the LeWiDi Shared Task (https://le-wi-di.github.io/).

def average_MD(targets, predictions):
    distances = []
    for tgt, pred in zip(targets, predictions):
        distance = sum(abs(p - t) for p, t in zip(pred, tgt))
        distances.append(distance)
    return round(np.mean(distances), 5) if distances else 0.0


def error_rate(targets, predictions):
    match_scores = []
    for tgt, pred in zip(targets, predictions):
        errors = sum(abs(t - p) for t, p in zip(tgt, pred))
        match_scores.append(1 - ((len(tgt) - errors) / len(tgt)))
    return float(np.mean(match_scores))
