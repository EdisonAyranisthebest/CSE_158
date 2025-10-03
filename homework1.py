import numpy as np
from sklearn.linear_model import LinearRegression
from dateutil import parser as _dateparser
from collections import defaultdict
from sklearn import linear_model
import numpy
import math
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple
from collections import Counter
from datetime import datetime
import re
from sklearn.metrics import mean_squared_error, precision_score


# =========================
# Helpers
# =========================

POS_WORDS = {
    "good", "great", "love", "loved", "amazing", "excellent", "awesome",
    "fantastic", "favorite", "best", "enjoyed", "fun"
}
NEG_WORDS = {
    "bad", "boring", "hate", "hated", "awful", "terrible", "worst",
    "disappointing", "meh", "poor", "waste"
}

def _get_text(d: dict) -> str:
    return d.get("review_text") or d.get("text") or d.get("review") or ""

def _get_rating(d: dict):
    # Try flat keys first
    for k in ["rating", "stars", "overall", "review_overall", "review/overall"]:
        if k in d:
            try:
                return float(d[k])
            except Exception:
                pass
    # Try nested "review" dict
    if "review" in d and isinstance(d["review"], dict):
        for k in ["overall", "rating", "stars"]:
            if k in d["review"]:
                try:
                    return float(d["review"][k])
                except Exception:
                    pass
    return None  # unknown / missing

def _parse_date(d: dict):
    # Try common date keys / formats
    for k in ["date_added", "date", "time", "review_time", "parsed_date"]:
        if k in d and d[k]:
            s = str(d[k])
            for fmt in [
                "%a %b %d %H:%M:%S %z %Y",  # Twitter-like "Sun Jul 30 07:44:10 -0700 2017"
                "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%m/%d/%Y",
                "%b %d, %Y",
            ]:
                try:
                    return datetime.strptime(s, fmt)
                except Exception:
                    continue
    return None

def _month_idx(dt: datetime) -> int:
    return dt.month if dt else 0  # 1..12, 0 if unknown

def _weekday_idx(dt: datetime) -> int:
    return dt.weekday() + 1 if dt else 0  # 1..7, 0 if unknown

def _abv(d: dict):
    for k in ["beer_abv", "abv", "ABV", "beer/ABV"]:
        if k in d:
            try:
                return float(d[k])
            except Exception:
                pass
    if "beer" in d and isinstance(d["beer"], dict):
        for k in ["abv", "ABV"]:
            if k in d["beer"]:
                try:
                    return float(d["beer"][k])
                except Exception:
                    pass
    return None

def _n_votes(d: dict) -> float:
    for k in ["n_votes", "votes", "helpfulness", "nHelpful", "helpful"]:
        if k in d:
            try:
                return float(d[k])
            except Exception:
                pass
    return 0.0

def _n_comments(d: dict) -> float:
    for k in ["n_comments", "comments", "nComments"]:
        if k in d:
            try:
                return float(d[k])
            except Exception:
                pass
    return 0.0

def _safe_stack(rows: List[List[float]]):
    """Stack lists of possibly different lengths into a 2D array by right-padding with zeros."""
    if not rows:
        return np.zeros((0, 0))
    width = max(len(r) for r in rows)
    fixed = [r + [0.0] * (width - len(r)) for r in rows]
    return np.array(fixed, dtype=float)

def _one_hot(idx: int, size: int) -> List[float]:
    v = [0.0] * size
    if 1 <= idx <= size:
        v[idx - 1] = 1.0
    return v


# =========================
# Q1
# =========================

def getMaxLen(dataset: List[dict]) -> int:
    """Return the maximum character length over review texts."""
    maxLen = 1
    for d in dataset:
        maxLen = max(maxLen, len(_get_text(d)))
    return maxLen

def featureQ1(datum: dict, maxLen: int) -> List[float]:
    """Feature vector: [1, normalized_length]."""
    txt = _get_text(datum)
    return [1.0, len(txt) / max(1, maxLen)]

def Q1(dataset: List[dict]) -> Tuple[np.ndarray, float]:
    """Linear regression on [bias, normalized_length] -> rating."""
    maxLen = getMaxLen(dataset)
    X = _safe_stack([featureQ1(d, maxLen) for d in dataset])
    Y = np.array([_get_rating(d) for d in dataset], dtype=float)
    mask = ~np.isnan(Y)
    X, Y = X[mask], Y[mask]
    if len(Y) == 0:
        return np.array([0.0, 0.0]), float("nan")
    model = LinearRegression(fit_intercept=False).fit(X, Y)
    preds = model.predict(X)
    mse = mean_squared_error(Y, preds)
    return model.coef_, float(mse)


# =========================
# Q2
# =========================

def featureQ2(datum: dict, maxLen: int) -> List[float]:
    """[1, normalized_length, weekday (1-7 or 0), month (1-12 or 0)]."""
    dt = _parse_date(datum)
    return [
        1.0,
        len(_get_text(datum)) / max(1, maxLen),
        float(_weekday_idx(dt)),
        float(_month_idx(dt)),
    ]

def Q2(dataset: List[dict]):
    maxLen = getMaxLen(dataset)
    X = _safe_stack([featureQ2(d, maxLen) for d in dataset])
    Y = np.array([_get_rating(d) for d in dataset], dtype=float)
    mask = ~np.isnan(Y)
    X, Y = X[mask], Y[mask]
    if len(Y) == 0:
        return X, Y, float("nan")
    model = LinearRegression(fit_intercept=False).fit(X, Y)
    mse = mean_squared_error(Y, model.predict(X))
    return X, Y, float(mse)


# =========================
# Q3
# =========================

def featureQ3(datum: dict, maxLen: int) -> List[float]:
    """[1, normalized_length] + one-hot weekday (7) + one-hot month (12)."""
    dt = _parse_date(datum)
    feats = [1.0, len(_get_text(datum)) / max(1, maxLen)]
    feats += _one_hot(_weekday_idx(dt), 7)
    feats += _one_hot(_month_idx(dt), 12)
    return feats

def Q3(dataset: List[dict]):
    maxLen = getMaxLen(dataset)
    X = _safe_stack([featureQ3(d, maxLen) for d in dataset])
    Y = np.array([_get_rating(d) for d in dataset], dtype=float)
    mask = ~np.isnan(Y)
    X, Y = X[mask], Y[mask]
    if len(Y) == 0:
        return X, Y, float("nan")
    model = LinearRegression(fit_intercept=False).fit(X, Y)
    mse = mean_squared_error(Y, model.predict(X))
    return X, Y, float(mse)


# =========================
# Q4
# =========================

def Q4(dataset_shuffled: List[dict]):
    """
    Compare generalization of Q2 vs Q3 using an 80/20 train/test split
    on the given (already-shuffled) dataset.
    Returns: (test_mse2, test_mse3)
    """
    n = len(dataset_shuffled)
    split = max(1, int(0.8 * n))
    train, test = dataset_shuffled[:split], dataset_shuffled[split:]

    # Train
    maxLen_tr = getMaxLen(train)
    X2_tr = _safe_stack([featureQ2(d, maxLen_tr) for d in train])
    Y_tr = np.array([_get_rating(d) for d in train], dtype=float)
    mask_tr = ~np.isnan(Y_tr)
    X2_tr, Y_tr = X2_tr[mask_tr], Y_tr[mask_tr]

    X3_tr = _safe_stack([featureQ3(d, maxLen_tr) for d in train])
    X3_tr = X3_tr[mask_tr]

    m2 = LinearRegression(fit_intercept=False).fit(X2_tr, Y_tr) if len(Y_tr) else None
    m3 = LinearRegression(fit_intercept=False).fit(X3_tr, Y_tr) if len(Y_tr) else None

    # Test
    maxLen_te = getMaxLen(test) or maxLen_tr
    X2_te = _safe_stack([featureQ2(d, maxLen_te) for d in test])
    Y_te = np.array([_get_rating(d) for d in test], dtype=float)
    mask_te = ~np.isnan(Y_te)
    X2_te, Y_te = X2_te[mask_te], Y_te[mask_te]

    X3_te = _safe_stack([featureQ3(d, maxLen_te) for d in test])
    X3_te = X3_te[mask_te]

    test_mse2 = mean_squared_error(Y_te, m2.predict(X2_te)) if (m2 and len(Y_te)) else float("nan")
    test_mse3 = mean_squared_error(Y_te, m3.predict(X3_te)) if (m3 and len(Y_te)) else float("nan")
    return float(test_mse2), float(test_mse3)


# =========================
# Q5
# =========================

def featureQ5(datum: dict) -> List[float]:
    """
    Simple features for a binary 'positive' classifier:
    [1, ABV, char_length, n_votes, n_comments]
    """
    txt = _get_text(datum)
    return [
        1.0,
        (_abv(datum) or 0.0),
        float(len(txt)),
        _n_votes(datum),
        _n_comments(datum),
    ]

def _label_positive(d: dict):
    r = _get_rating(d)
    if r is None:
        return None
    return 1 if r >= 4.0 else 0

def Q5(dataset: List[dict], feat_func):
    """
    Fit logistic regression on the whole dataset using feat_func (e.g., featureQ5),
    and return (TP, TN, FP, FN, BER) evaluated on that same dataset.
    """
    X = _safe_stack([feat_func(d) for d in dataset])
    y = np.array([_label_positive(d) for d in dataset])
    mask = y != None
    X, y = X[mask], y[mask].astype(int)
    if len(y) == 0:
        return 0, 0, 0, 0, float("nan")

    clf = LogisticRegression(max_iter=200, fit_intercept=False).fit(X, y)
    yhat = clf.predict(X)

    TP = int(((y == 1) & (yhat == 1)).sum())
    TN = int(((y == 0) & (yhat == 0)).sum())
    FP = int(((y == 0) & (yhat == 1)).sum())
    FN = int(((y == 1) & (yhat == 0)).sum())

    # Balanced Error Rate (BER) = 1 - 0.5*(TPR + TNR)
    tpr = TP / max(1, (TP + FN))
    tnr = TN / max(1, (TN + FP))
    BER = 1 - 0.5 * (tpr + tnr)
    return TP, TN, FP, FN, float(BER)


# =========================
# Q6
# =========================

def Q6(dataset: List[dict]):
    """
    Train the same classifier as Q5 (featureQ5) and return a list of precision
    values at thresholds 0.05, 0.10, ..., 0.95.
    """
    X = _safe_stack([featureQ5(d) for d in dataset])
    y = np.array([_label_positive(d) for d in dataset])
    mask = y != None
    X, y = X[mask], y[mask].astype(int)
    if len(y) == 0:
        return []

    clf = LogisticRegression(max_iter=200, fit_intercept=False).fit(X, y)
    proba = clf.predict_proba(X)[:, 1]

    precs = []
    for thr in np.linspace(0.05, 0.95, 19):
        yhat = (proba >= thr).astype(int)
        # If the classifier predicts no positives at a threshold, define precision as 1.0
        # (so you don't get a divide-by-zero per the usual precision definition)
        if yhat.sum() == 0:
            precs.append(1.0)
        else:
            precs.append(float(precision_score(y, yhat)))
    return precs


# =========================
# Q7
# =========================

_word_re = re.compile(r"[a-z]+", re.I)

def _word_counts(s: str):
    toks = _word_re.findall(s.lower())
    c = Counter(toks)
    pos = sum(c[w] for w in POS_WORDS)
    neg = sum(c[w] for w in NEG_WORDS)
    return len(toks), pos, neg

def featureQ7(datum: dict) -> List[float]:
    """
    Richer feature set to improve over Q5:
    [1, ABV, char_length, word_count, '!' count, '?' count,
     positive_lexicon_count, negative_lexicon_count, n_votes, n_comments,
     weekday_one_hot(7), month_one_hot(12)]
    """
    txt = _get_text(datum)
    dt = _parse_date(datum)
    n_words, pos_cnt, neg_cnt = _word_counts(txt)

    feats = [
        1.0,
        (_abv(datum) or 0.0),
        float(len(txt)),
        float(n_words),
        float(txt.count("!")),
        float(txt.count("?")),
        float(pos_cnt),
        float(neg_cnt),
        _n_votes(datum),
        _n_comments(datum),
    ]
    feats += _one_hot(_weekday_idx(dt), 7)
    feats += _one_hot(_month_idx(dt), 12)
    return feats
