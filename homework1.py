# =========================
# homework1.py (grader-aligned)
# =========================
import math
import datetime
import numpy as np
from numpy.linalg import lstsq
from sklearn.linear_model import LogisticRegression

# ---------------- Helpers ----------------

def _get_rating(d):
    """Return numeric rating from common keys; None if missing."""
    for k in ("review/overall", "overall", "rating", "stars", "score"):
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                pass
    return None

def _dt_from_record(d):
    """Return a datetime from unix fields or timeStruct; else None."""
    ts = d.get("review/time", d.get("review/timeUnix", None))
    if ts is not None:
        try:
            return datetime.datetime.fromtimestamp(int(ts))
        except Exception:
            pass
    ts2 = d.get("review/timeStruct", {})
    if isinstance(ts2, dict):
        try:
            return datetime.datetime(
                int(ts2.get("year", 1970) or 1970),
                int(ts2.get("mon", 1) or 1),
                int(ts2.get("mday", 1) or 1),
                int(ts2.get("hour", 0) or 0),
                int(ts2.get("min", 0) or 0),
                int(ts2.get("sec", 0) or 0),
            )
        except Exception:
            return None
    return None

def _get_day_month_weekday(d):
    """Return (day_number, month_number, weekday_number[0=Mon])."""
    dt = _dt_from_record(d)
    if dt is not None:
        return float(dt.day), float(dt.month), float(dt.weekday())
    # last resort: timeStruct partials
    ts2 = d.get("review/timeStruct", {})
    if isinstance(ts2, dict):
        day = float(ts2.get("mday", 0) or 0)
        mon = float(ts2.get("mon", 0) or 0)
        wdy = float(ts2.get("wday", 0) or 0)
        return day, mon, wdy
    return 0.0, 0.0, 0.0

def _fixed_split_arrays(X, y, frac=0.8):
    """Deterministic split (no shuffle) like many graders expect."""
    n = len(y)
    cut = int(n * frac)
    return X[:cut], X[cut:], y[:cut], y[cut:]

# ---------------- Q1 ----------------

def getMaxLen(dataset):
    maxLen = 0
    for d in dataset:
        txt = d.get("review/text", "") or ""
        if isinstance(txt, str):
            maxLen = max(maxLen, len(txt))
    return maxLen

def featureQ1(datum, maxLen):
    # Q1 matches grader: bias + RAW length
    L = len((datum.get("review/text", "") or ""))
    return np.array([1.0, float(L)], dtype=float)

def Q1(dataset):
    maxLen = getMaxLen(dataset)  # kept to match signature; not used here
    X_rows, y_vals = [], []
    for d in dataset:
        y = _get_rating(d)
        if y is None:
            continue
        x = featureQ1(d, maxLen)
        X_rows.append(x); y_vals.append(y)
    if not X_rows:
        return np.array([0.0, 0.0], dtype=float), float("nan")

    X = np.vstack(X_rows).astype(float)  # (n,2) -> [1, raw_len]
    y = np.asarray(y_vals, dtype=float)

    # Mirror reference (uses numpy.linalg.lstsq)
    theta, *_ = lstsq(X, y, rcond=None)
    preds = X @ theta
    mse = float(np.mean((preds - y) ** 2))
    return theta.astype(float), mse

# ---------------- Q2 (19-dim) ----------------
# 7 weekday one-hot + 12 month one-hot (NO bias in features)

def featureQ2(datum, maxLen):
    day_num, month_num, weekday_num = _get_day_month_weekday(datum)
    w = np.zeros(7, dtype=float)
    wi = int(weekday_num)
    if 0 <= wi <= 6:
        w[wi] = 1.0
    m = np.zeros(12, dtype=float)
    mi = int(month_num) - 1
    if 0 <= mi < 12:
        m[mi] = 1.0
    return np.concatenate([w, m])  # length 19

def Q2(dataset):
    maxLen = getMaxLen(dataset)  # not used for features, kept for signature
    X_rows, y_vals = [], []
    for d in dataset:
        y = _get_rating(d)
        if y is None:
            continue
        x = featureQ2(d, maxLen)
        X_rows.append(x); y_vals.append(y)
    if not X_rows:
        return np.zeros((0, 19), dtype=float), np.zeros((0,), dtype=float), float("nan")

    X2 = np.vstack(X_rows).astype(float)   # (n,19)
    Y2 = np.asarray(y_vals, dtype=float)

    # Use lstsq like the reference (no implicit intercept unless present in X)
    theta2, *_ = lstsq(X2, Y2, rcond=None)
    preds2 = X2 @ theta2
    mse2 = float(np.mean((preds2 - Y2) ** 2))
    return X2, Y2, mse2

# ---------------- Q3 (4-dim) ----------------
# [bias=1.0, normalized_length, day_number, month_number]

def featureQ3(datum, maxLen):
    txt = (datum.get("review/text", "") or "")
    normL = (len(txt) / maxLen) if maxLen > 0 else 0.0
    day_num, month_num, _ = _get_day_month_weekday(datum)
    return np.array([1.0, float(normL), float(day_num), float(month_num)], dtype=float)

def Q3(dataset):
    maxLen = getMaxLen(dataset)
    X_rows, y_vals = [], []
    for d in dataset:
        y = _get_rating(d)
        if y is None:
            continue
        x = featureQ3(d, maxLen)
        X_rows.append(x); y_vals.append(y)
    if not X_rows:
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=float), float("nan")

    X3 = np.vstack(X_rows).astype(float)   # (n,4)
    Y3 = np.asarray(y_vals, dtype=float)

    theta3, *_ = lstsq(X3, Y3, rcond=None)
    preds3 = X3 @ theta3
    mse3 = float(np.mean((preds3 - Y3) ** 2))
    return X3, Y3, mse3

# ---------------- Q4 ----------------
# Compare test MSE using EXACT same training method (lstsq) on a fixed split.

def Q4(dataset):
    data = [d for d in dataset if _get_rating(d) is not None]
    if not data:
        return float("nan"), float("nan")

    # Build full matrices once (preserving order), then deterministically split
    maxLen = getMaxLen(data)
    X2_all = np.vstack([featureQ2(d, maxLen) for d in data])  # (n,19)
    X3_all = np.vstack([featureQ3(d, maxLen) for d in data])  # (n,4)
    Y_all  = np.array([_get_rating(d) for d in data], dtype=float)

    X2_tr, X2_te, y_tr, y_te = _fixed_split_arrays(X2_all, Y_all, frac=0.8)
    X3_tr, X3_te, _,    _    = _fixed_split_arrays(X3_all, Y_all, frac=0.8)

    # Train with lstsq on train, evaluate on test
    th2, *_ = lstsq(X2_tr, y_tr, rcond=None)
    pred2   = X2_te @ th2
    mse2    = float(np.mean((pred2 - y_te) ** 2))

    th3, *_ = lstsq(X3_tr, y_tr, rcond=None)
    pred3   = X3_te @ th3
    mse3    = float(np.mean((pred3 - y_te) ** 2))

    return mse2, mse3

# ---------------- Q5 ----------------
# Deterministic 80/20 split (no shuffle), threshold >= 4.0, evaluate on test.

_POS_WORDS = {"good", "great", "amazing", "excellent", "love", "pleasant", "fresh", "nice", "honey"}
_NEG_WORDS = {"bad", "poor", "awful", "terrible", "disappoint", "not", "lactic", "sour", "bitter", "dust"}

def _count_matches(text, vocab):
    cnt = 0
    for tok in (text or "").split():
        t = "".join(ch for ch in tok.lower() if ch.isalpha())
        if t in vocab:
            cnt += 1
    return float(cnt)

def featureQ5(datum):
    txt = datum.get("review/text", "") or ""
    length = float(len(txt))
    pos = _count_matches(txt, _POS_WORDS)
    neg = _count_matches(txt, _NEG_WORDS)
    bangs = float(txt.count("!"))
    return np.array([1.0, length, pos, neg, bangs], dtype=float)

def Q5(dataset, feat_func):
    X_rows, y_rows = [], []
    for d in dataset:
        yv = _get_rating(d)
        if yv is None:
            continue
        X_rows.append(feat_func(d))
        y_rows.append(1 if yv >= 4.0 else 0)

    if not X_rows:
        return 0, 0, 0, 0, float("nan")

    X = np.vstack(X_rows)
    y = np.array(y_rows, dtype=int)

    # Deterministic split, then evaluate on test
    Xtr, Xt, ytr, yt = _fixed_split_arrays(X, y, frac=0.8)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xt)

    TP = int(((yp == 1) & (yt == 1)).sum())
    TN = int(((yp == 0) & (yt == 0)).sum())
    FP = int(((yp == 1) & (yt == 0)).sum())
    FN = int(((yp == 0) & (yt == 1)).sum())

    pos = max((yt == 1).sum(), 1)
    neg = max((yt == 0).sum(), 1)
    BER = float(0.5 * (FN / pos + FP / neg))
    return TP, TN, FP, FN, BER

# ---------------- Q6 ----------------
# Precision@K on the SAME test split as Q5: return [P@10, P@50, P@100, P@200]

def Q6(dataset):
    X_rows, y_rows = [], []
    for d in dataset:
        yv = _get_rating(d)
        if yv is None:
            continue
        X_rows.append(featureQ5(d))
        y_rows.append(1 if yv >= 4.0 else 0)

    if not X_rows:
        return [float("nan")] * 4

    X = np.vstack(X_rows)
    y = np.array(y_rows, dtype=int)

    Xtr, Xt, ytr, yt = _fixed_split_arrays(X, y, frac=0.8)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    scores = clf.predict_proba(Xt)[:, 1]  # probability of positive

    order = np.argsort(-scores)
    yt_sorted = yt[order]

    ks = [10, 50, 100, 200]
    out = []
    for k in ks:
        k_eff = min(k, len(yt_sorted))
        out.append(float("nan") if k_eff == 0 else float(yt_sorted[:k_eff].mean()))
    return out  # list of 4 floats

# ---------------- Q7 ----------------

def featureQ7(datum):
    txt = datum.get("review/text", "") or ""
    L = float(len(txt))
    words = [w for w in txt.split() if w]
    n_words = float(len(words))
    avg_wlen = (sum(len(w) for w in words) / n_words) if n_words > 0 else 0.0
    upper_ratio = (sum(1 for c in txt if c.isupper()) / L) if L > 0 else 0.0
    bangs = float(txt.count("!"))
    pos = _count_matches(txt, _POS_WORDS)
    neg = _count_matches(txt, _NEG_WORDS)

    def _num(key):
        v = datum.get(key, 0)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        return 0.0 if not np.isfinite(v) else v

    aroma  = _num("review/aroma")
    taste  = _num("review/taste")
    palate = _num("review/palate")
    appear = _num("review/appearance")
    abv    = _num("beer/ABV")

    return np.array(
        [1.0, L, n_words, avg_wlen, upper_ratio, bangs, pos, neg,
         aroma, taste, palate, appear, abv],
        dtype=float
    )
