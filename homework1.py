# =========================
# homework1.py (final aligned)
# =========================
import math
import datetime
import numpy as np
from sklearn import linear_model
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
    """Return a datetime from unix keys if possible, else None."""
    ts = d.get("review/time", d.get("review/timeUnix", None))
    if ts is not None:
        try:
            return datetime.datetime.fromtimestamp(int(ts))
        except Exception:
            pass
    # fallback to timeStruct if present
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
            pass
    return None

def _get_day_month_weekday(d):
    """Return (day_number, month_number, weekday_number[0=Mon])."""
    dt = _dt_from_record(d)
    if dt is not None:
        return float(dt.day), float(dt.month), float(dt.weekday())
    # last resort
    ts2 = d.get("review/timeStruct", {})
    if isinstance(ts2, dict):
        day = float(ts2.get("mday", 0) or 0)
        mon = float(ts2.get("mon", 0) or 0)
        wdy = float(ts2.get("wday", 0) or 0)
        return day, mon, wdy
    return 0.0, 0.0, 0.0

# ---------------- Q1 ----------------

def getMaxLen(dataset):
    maxLen = 0
    for d in dataset:
        txt = d.get("review/text", "") or ""
        if isinstance(txt, str):
            maxLen = max(maxLen, len(txt))
    return maxLen

def featureQ1(datum, maxLen):
    txt = datum.get("review/text", "") or ""
    L = len(txt)
    normL = (L / maxLen) if maxLen > 0 else 0.0
    return np.array([1.0, float(normL)], dtype=float)

def Q1(dataset):
    maxLen = getMaxLen(dataset)
    X_rows, y_vals = [], []
    for d in dataset:
        y = _get_rating(d)
        if y is None:
            continue
        x = featureQ1(d, maxLen)
        if np.isfinite(y) and np.all(np.isfinite(x)):
            X_rows.append(x); y_vals.append(y)

    if not X_rows:
        return np.array([0.0, 0.0], dtype=float), float("nan")

    X = np.vstack(X_rows).astype(float)   # shape (n,2) -> [1, norm_len]
    y = np.asarray(y_vals, dtype=float)

    model = linear_model.LinearRegression(fit_intercept=False)  # bias in features
    model.fit(X, y)
    theta = model.coef_.astype(float)
    mse = float(np.mean((model.predict(X) - y) ** 2))
    return theta, mse

# ---------------- Q2 (19-dim) ----------------
# 7 one-hot weekday + 12 one-hot month

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
    maxLen = getMaxLen(dataset)  # not used but kept for signature
    X_rows, y_vals = [], []
    for d in dataset:
        y = _get_rating(d)
        if y is None:
            continue
        x = featureQ2(d, maxLen)
        if np.isfinite(y) and np.all(np.isfinite(x)):
            X_rows.append(x); y_vals.append(y)

    if not X_rows:
        return np.zeros((0, 19), dtype=float), np.zeros((0,), dtype=float), float("nan")

    X2 = np.vstack(X_rows).astype(float)   # (n,19)
    Y2 = np.asarray(y_vals, dtype=float)

    model = linear_model.LinearRegression(fit_intercept=True)  # add intercept
    model.fit(X2, Y2)
    mse2 = float(np.mean((model.predict(X2) - Y2) ** 2))
    return X2, Y2, mse2

# ---------------- Q3 (4-dim) ----------------
# [1.0, normalized_length, day_number, month_number]

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
        if np.isfinite(y) and np.all(np.isfinite(x)):
            X_rows.append(x); y_vals.append(y)

    if not X_rows:
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=float), float("nan")

    X3 = np.vstack(X_rows).astype(float)   # (n,4)
    Y3 = np.asarray(y_vals, dtype=float)

    model = linear_model.LinearRegression(fit_intercept=False)  # bias included
    model.fit(X3, Y3)
    mse3 = float(np.mean((model.predict(X3) - Y3) ** 2))
    return X3, Y3, mse3

# ---------------- Q4 ----------------
# Deterministic comparison using identical order split

def Q4(dataset):
    data = [d for d in dataset if _get_rating(d) is not None]
    if not data:
        return float("nan"), float("nan")

    maxLen = getMaxLen(data)
    X2_all = np.vstack([featureQ2(d, maxLen) for d in data])  # (n,19)
    X3_all = np.vstack([featureQ3(d, maxLen) for d in data])  # (n,4)
    Y_all  = np.array([_get_rating(d) for d in data], dtype=float)

    n = len(Y_all)
    cut = int(0.8 * n)
    def _mse(X):
        reg = linear_model.LinearRegression(fit_intercept=(X.shape[1] != 4))
        if X.shape[1] == 4:
            reg.set_params(fit_intercept=False)
        reg.fit(X[:cut], Y_all[:cut])
        pred = reg.predict(X[cut:])
        return float(np.mean((pred - Y_all[cut:]) ** 2))

    return _mse(X2_all), _mse(X3_all)

# ---------------- Q5 ----------------
# Train & evaluate on FULL dataset; positive if rating >= 4.0

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
    X_rows, y_vals = [], []
    for d in dataset:
        y = _get_rating(d)
        if y is None:
            continue
        X_rows.append(feat_func(d))
        y_vals.append(1 if y >= 4.0 else 0)  # grader-style threshold

    if not X_rows:
        return 0, 0, 0, 0, float("nan")

    X = np.vstack(X_rows)
    y = np.array(y_vals, dtype=int)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)
    yp = clf.predict(X)  # evaluate on full set

    TP = int(((yp == 1) & (y == 1)).sum())
    TN = int(((yp == 0) & (y == 0)).sum())
    FP = int(((yp == 1) & (y == 0)).sum())
    FN = int(((yp == 0) & (y == 1)).sum())

    pos = max((y == 1).sum(), 1)
    neg = max((y == 0).sum(), 1)
    BER = float(0.5 * (FN / pos + FP / neg))
    return TP, TN, FP, FN, BER

# ---------------- Q6 ----------------
# Precision@K on FULL dataset (same model style as Q5)

def Q6(dataset):
    X_rows, y_vals = [], []
    for d in dataset:
        y = _get_rating(d)
        if y is None:
            continue
        X_rows.append(featureQ5(d))
        y_vals.append(1 if y >= 4.0 else 0)

    if not X_rows:
        return [float("nan")] * 4

    X = np.vstack(X_rows)
    y = np.array(y_vals, dtype=int)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)
    scores = clf.predict_proba(X)[:, 1]  # on full set

    order = np.argsort(-scores)
    yt_sorted = y[order]

    ks = [10, 50, 100, 200]
    out = []
    for k in ks:
        k_eff = min(k, len(yt_sorted))
        out.append(float("nan") if k_eff == 0 else float(yt_sorted[:k_eff].mean()))
    return out  # [P@10, P@50, P@100, P@200]

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
