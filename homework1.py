# =========================
# homework1.py  (corrected)
# =========================
import math
import datetime
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------- helpers ----------

def _get_rating(d):
    """Return numeric rating from common keys; None if missing."""
    for k in ("review/overall", "overall", "rating", "stars", "score"):
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                pass
    return None

def _get_day_month(datum):
    """Extract (day, month) from several time formats; fallback (0,0)."""
    ts = datum.get("review/time", datum.get("review/timeUnix", None))
    if ts is not None:
        try:
            dt = datetime.datetime.fromtimestamp(int(ts))
            return float(dt.day), float(dt.month)
        except Exception:
            pass
    ts2 = datum.get("review/timeStruct", {})
    if isinstance(ts2, dict):
        day = float(ts2.get("mday", 0) or 0)
        mon = float(ts2.get("mon", 0) or 0)
        return day, mon
    return 0.0, 0.0

# ---------- Q1 ----------

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
    return np.array([1.0, normL], dtype=float)

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

    X = np.vstack(X_rows).astype(float)
    y = np.asarray(y_vals, dtype=float)
    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(X, y)
    theta = model.coef_.astype(float)
    mse = float(np.mean((model.predict(X) - y) ** 2))
    return theta, mse

# ---------- Q2 ----------

def featureQ2(datum, maxLen):
    txt = datum.get("review/text", "") or ""
    L = len(txt)
    normL = (L / maxLen) if maxLen > 0 else 0.0
    day, month = _get_day_month(datum)
    return np.array([1.0, normL, day, month], dtype=float)

def Q2(dataset):
    maxLen = getMaxLen(dataset)
    X_rows, y_vals = [], []
    for d in dataset:
        y = _get_rating(d)
        if y is None:
            continue
        x = featureQ2(d, maxLen)
        if np.isfinite(y) and np.all(np.isfinite(x)):
            X_rows.append(x); y_vals.append(y)

    if not X_rows:
        # keep shapes valid (0x4 matrix, 0-length y)
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=float), float("nan")

    X2 = np.vstack(X_rows).astype(float)
    Y2 = np.asarray(y_vals, dtype=float)
    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(X2, Y2)
    mse2 = float(np.mean((model.predict(X2) - Y2) ** 2))
    return X2, Y2, mse2

# ---------- Q3 ----------

def featureQ3(datum, maxLen, user_bucket_size=100):
    base = featureQ2(datum, maxLen)  # [1, normL, day, month]
    user = str(datum.get("user/profileName", "") or "")
    user_vec = np.zeros(user_bucket_size, dtype=float)
    if user:
        user_vec[hash(user) % user_bucket_size] = 1.0
    return np.concatenate([base, user_vec])

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
        return np.zeros((0, 4 + 100), dtype=float), np.zeros((0,), dtype=float), float("nan")

    X3 = np.vstack(X_rows).astype(float)
    Y3 = np.asarray(y_vals, dtype=float)
    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(X3, Y3)
    mse3 = float(np.mean((model.predict(X3) - Y3) ** 2))
    return X3, Y3, mse3

# ---------- Q4 ----------

def Q4(dataset):
    data = [d for d in dataset if _get_rating(d) is not None]
    if not data:
        return float("nan"), float("nan")

    train, test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
    maxLen_train = getMaxLen(train)

    # Q2 encoding
    Xtr2 = np.vstack([featureQ2(d, maxLen_train) for d in train])
    ytr  = np.array([_get_rating(d) for d in train], dtype=float)
    Xt2  = np.vstack([featureQ2(d, maxLen_train) for d in test])
    yt   = np.array([_get_rating(d) for d in test], dtype=float)
    reg2 = linear_model.LinearRegression(fit_intercept=False)
    reg2.fit(Xtr2, ytr)
    test_mse2 = float(np.mean((reg2.predict(Xt2) - yt) ** 2))

    # Q3 encoding
    Xtr3 = np.vstack([featureQ3(d, maxLen_train) for d in train])
    Xt3  = np.vstack([featureQ3(d, maxLen_train) for d in test])
    reg3 = linear_model.LinearRegression(fit_intercept=False)
    reg3.fit(Xtr3, ytr)
    test_mse3 = float(np.mean((reg3.predict(Xt3) - yt) ** 2))

    return test_mse2, test_mse3

# ---------- Q5 ----------

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
    rows = []
    for d in dataset:
        y = _get_rating(d)
        if y is None:
            continue
        x = feat_func(d)
        if x is None or not np.all(np.isfinite(x)):
            continue
        rows.append((x, 1 if y >= 4.0 else 0))

    if not rows:
        return 0, 0, 0, 0, float("nan")

    X = np.vstack([r[0] for r in rows])
    y = np.array([r[1] for r in rows], dtype=int)

    Xtr, Xt, ytr, yt = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    clf = LogisticRegression(max_iter=1000)
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

# ---------- Q6 ----------

def Q6(dataset):
    rows = []
    for d in dataset:
        y = _get_rating(d)
        if y is None:
            continue
        rows.append((featureQ5(d), 1 if y >= 4.0 else 0))

    if not rows:
        return [float("nan")] * 4

    X = np.vstack([r[0] for r in rows])
    y = np.array([r[1] for r in rows], dtype=int)

    Xtr, Xt, ytr, yt = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    scores = clf.predict_proba(Xt)[:, 1]

    order = np.argsort(-scores)
    yt_sorted = yt[order]

    ks = [10, 50, 100, 200]
    out = []
    for k in ks:
        k_eff = min(k, len(yt_sorted))
        out.append(float("nan") if k_eff == 0 else float(yt_sorted[:k_eff].mean()))
    return out  # list of 4 floats

# ---------- Q7 ----------

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
