import numpy as np
from numpy.linalg import lstsq
from sklearn.linear_model import LogisticRegression
import datetime

# ---------------- helpers ----------------

def _get_text(d):
    return str(
        d.get("review/text")
        or d.get("reviewText")
        or d.get("review_text")
        or d.get("text")
        or ""
    )

def _get_rating(d):
    for k in ("review/overall", "overall", "rating", "stars", "score"):
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                pass
    rv = d.get("review")
    if isinstance(rv, dict) and "overall" in rv:
        try:
            return float(rv["overall"])
        except Exception:
            return None
    return None

def _get_day_month_weekday(d):
    """
    Prefer review/timeStruct if present (matches dataset's own weekday/month),
    otherwise fall back to UNIX timestamps.
    Returns (day, month, weekday) where weekday is 0..6 (Mon..Sun).
    """
    ts = d.get("review/timeStruct")
    if isinstance(ts, dict):
        try:
            day = float(ts.get("mday", 0) or 0)
            mon = float(ts.get("mon", 0) or 0)
            wdy = float(ts.get("wday", 0) or 0)  # dataset uses 0=Mon
            return day, mon, wdy
        except Exception:
            pass

    for k in ("review/timeUnix", "review/time"):
        if k in d and d[k] is not None:
            try:
                dt = datetime.datetime.fromtimestamp(int(d[k]))
                return float(dt.day), float(dt.month), float(dt.weekday())  # 0=Mon..6=Sun
            except Exception:
                pass
    return 0.0, 0.0, 0.0

def getMaxLen(dataset):
    m = 0
    for d in (dataset or []):
        m = max(m, len(_get_text(d)))
    return m

# ---------------- Q1 ----------------

def featureQ1(datum, maxLen):
    s = _get_text(datum)
    norm_len = (len(s) / maxLen) if maxLen > 0 else 0.0
    return np.array([1.0, float(norm_len)], dtype=float)

def Q1(dataset):
    maxLen = getMaxLen(dataset)
    X, y = [], []
    for d in (dataset or []):
        r = _get_rating(d)
        if r is None: 
            continue
        X.append(featureQ1(d, maxLen)); y.append(r)
    if not X:
        return np.array([0.0, 0.0], dtype=float), float("nan")
    X = np.vstack(X).astype(float); y = np.asarray(y, dtype=float)
    theta, *_ = lstsq(X, y, rcond=None)
    mse = float(np.mean((X @ theta - y) ** 2))
    return theta.astype(float), mse

# ---------------- Q2 (19-dim) ----------------
# weekday one-hot (7) + month one-hot (12), no bias/length

def featureQ2(datum, maxLen):
    _, month_num, weekday_num = _get_day_month_weekday(datum)

    w = np.zeros(7, dtype=float)     # 0..6 Mon..Sun
    wi = int(weekday_num)
    if 0 <= wi <= 6:
        w[wi] = 1.0

    m = np.zeros(12, dtype=float)    # 1..12 -> 0..11 Jan..Dec
    mi = int(month_num) - 1
    if 0 <= mi < 12:
        m[mi] = 1.0

    return np.concatenate([w, m]).astype(float)  # length 19

def Q2(dataset):
    maxLen = getMaxLen(dataset)
    X, Y = [], []
    for d in (dataset or []):
        r = _get_rating(d)
        if r is None: 
            continue
        X.append(featureQ2(d, maxLen)); Y.append(r)
    X2 = np.vstack(X) if X else np.zeros((0, 19), dtype=float)
    Y2 = np.asarray(Y, dtype=float)
    if Y2.size == 0:
        return X2, Y2, float("nan")
    theta2, *_ = lstsq(X2, Y2, rcond=None)
    MSE2 = float(np.mean((X2 @ theta2 - Y2) ** 2))
    return X2, Y2, MSE2

# ---------------- Q3 (4-dim) ----------------
# [1.0, RAW review length, weekday_number (0..6), month_number (1..12)]

def featureQ3(datum, maxLen):
    raw_len = float(len(_get_text(datum)))
    _, month_num, weekday_num = _get_day_month_weekday(datum)
    return np.array([1.0, raw_len, float(weekday_num), float(month_num)], dtype=float)

def Q3(dataset):
    maxLen = getMaxLen(dataset)
    X, Y = [], []
    for d in (dataset or []):
        r = _get_rating(d)
        if r is None: 
            continue
        X.append(featureQ3(d, maxLen)); Y.append(r)
    X3 = np.vstack(X) if X else np.zeros((0, 4), dtype=float)
    Y3 = np.asarray(Y, dtype=float)
    if Y3.size == 0:
        return X3, Y3, float("nan")
    theta3, *_ = lstsq(X3, Y3, rcond=None)
    MSE3 = float(np.mean((X3 @ theta3 - Y3) ** 2))
    return X3, Y3, MSE3

# ---------------- Q4 ----------------

def Q4(dataset):
    data = [d for d in (dataset or []) if _get_rating(d) is not None]
    if not data:
        return float("nan"), float("nan")
    maxLen = getMaxLen(data)
    X2_all = np.vstack([featureQ2(d, maxLen) for d in data])
    X3_all = np.vstack([featureQ3(d, maxLen) for d in data])
    Y_all  = np.array([_get_rating(d) for d in data], dtype=float)

    n = len(Y_all); cut = int(0.8 * n)
    X2_tr, X2_te, y_tr, y_te = X2_all[:cut], X2_all[cut:], Y_all[:cut], Y_all[cut:]
    X3_tr, X3_te = X3_all[:cut], X3_all[cut:]

    th2, *_ = lstsq(X2_tr, y_tr, rcond=None)
    th3, *_ = lstsq(X3_tr, y_tr, rcond=None)

    mse2 = float(np.mean((X2_te @ th2 - y_te) ** 2))
    mse3 = float(np.mean((X3_te @ th3 - y_te) ** 2))
    return mse2, mse3

# ---------------- Q5 / Q6 / Q7 ----------------

def featureQ5(datum):
    return np.array([1.0, float(len(_get_text(datum)))], dtype=float)

def _label_ge4(d):
    r = _get_rating(d)
    if r is None: 
        return None
    return 1 if r >= 4.0 else 0

def Q5(dataset, feat_func):
    Xrows, yrows = [], []
    for d in (dataset or []):
        lab = _label_ge4(d)
        if lab is None:
            continue
        Xrows.append(feat_func(d)); yrows.append(lab)
    if not Xrows:
        return 0, 0, 0, 0, float('nan')

    X = np.vstack(Xrows).astype(float)
    y = np.asarray(yrows, dtype=int)

    clf = LogisticRegression(max_iter=2000, class_weight='balanced', fit_intercept=False)
    clf.fit(X, y)
    yp = clf.predict(X)

    TP = int(((yp == 1) & (y == 1)).sum())
    TN = int(((yp == 0) & (y == 0)).sum())
    FP = int(((yp == 1) & (y == 0)).sum())
    FN = int(((yp == 0) & (y == 1)).sum())

    P = max(int((y == 1).sum()), 1)
    N = max(int((y == 0).sum()), 1)
    BER = float(0.5 * (FN / P + FP / N))
    return TP, TN, FP, FN, BER

def Q6(dataset):
    # Same data/model as Q5 baseline; return a DICT of precisions
    Xrows, yrows = [], []
    for d in (dataset or []):
        lab = _label_ge4(d)
        if lab is None:
            continue
        Xrows.append(featureQ5(d)); yrows.append(lab)
    if not Xrows:
        return {10: float('nan'), 50: float('nan'), 100: float('nan'), 200: float('nan')}

    X = np.vstack(Xrows).astype(float)
    y = np.asarray(yrows, dtype=int)

    clf = LogisticRegression(max_iter=2000, class_weight='balanced', fit_intercept=False)
    clf.fit(X, y)
    scores = clf.predict_proba(X)[:, 1]
    order = np.argsort(-scores)
    yt_sorted = y[order]

    out = {}
    for K in [10, 50, 100, 200]:
        k = min(K, len(yt_sorted))
        out[K] = float('nan') if k == 0 else float(yt_sorted[:k].mean())
    return out

def featureQ7(datum):
    s = _get_text(datum)
    toks = [t.strip(".,!?;:()[]{}'\"").lower() for t in s.split() if t]
    pos = {"good","great","excellent","amazing","love","awesome","fantastic",
           "perfect","best","wonderful","favorite","happy","tasty","fresh",
           "crisp","smooth","aroma","balanced","complex"}
    neg = {"bad","terrible","awful","hate","worst","poor","disappointing",
           "boring","flat","stale","skunky","bitter","sour","thin","watery",
           "metallic","off"}
    pos_cnt = float(sum(t in pos for t in toks))
    neg_cnt = float(sum(t in neg for t in toks))
    length  = float(len(s))
    bangs   = float(s.count('!'))
    qmarks  = float(s.count('?'))
    digits  = float(sum(ch.isdigit() for ch in s))
    caps_ratio = (sum(1 for ch in s if ch.isalpha() and ch.isupper()) / (1.0 + len(s)))
    avg_wlen = (sum(len(t) for t in toks) / len(toks)) if toks else 0.0
    return np.array([1.0, length, pos_cnt, neg_cnt, bangs, qmarks, avg_wlen, digits, float(caps_ratio)], dtype=float)

def Q7(dataset):
    _, _, _, _, BER5 = Q5(dataset, featureQ5)
    _, _, _, _, BER7 = Q5(dataset, featureQ7)
    return BER5, BER7
