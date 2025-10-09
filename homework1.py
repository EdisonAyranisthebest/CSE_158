import numpy as np
from numpy.linalg import lstsq
from sklearn.linear_model import LogisticRegression
import datetime

# try to use dateutil for robust string parsing (falls back cleanly if missing)
try:
    from dateutil import parser as _dateparser
except Exception:
    _dateparser = None

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

def _parse_string_time(s):
    """Best-effort parse for string timestamps."""
    if s is None:
        return None
    # unix given as a string?
    try:
        if str(s).isdigit():
            return datetime.datetime.fromtimestamp(int(s))
    except Exception:
        pass
    # general string like '02/16/2009' etc.
    if _dateparser is not None:
        try:
            return _dateparser.parse(str(s))
        except Exception:
            return None
    # minimal fallback: try common formats
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d-%b-%Y"):
        try:
            return datetime.datetime.strptime(str(s), fmt)
        except Exception:
            pass
    return None

def _get_day_month_weekday(d):
    """
    Return (day, month, weekday) with weekday 0..6 (Mon..Sun) and month 1..12.
    Prefer review/timeStruct; fall back to UNIX then string review/time.
    """
    ts = d.get("review/timeStruct")
    if isinstance(ts, dict):
        try:
            day = int(ts.get("mday", 0) or 0)
            mon = int(ts.get("mon", 0) or 0)
            wdy = int(ts.get("wday", 0) or 0)  # 0=Mon..6=Sun (matches sample)
            return day, mon, wdy
        except Exception:
            pass

    for k in ("review/timeUnix", "review/time"):
        if k in d and d[k] is not None:
            # unix?
            try:
                dt = datetime.datetime.fromtimestamp(int(d[k]))
                return dt.day, dt.month, dt.weekday()
            except Exception:
                # maybe a string time (e.g., '02/16/2009')
                dt = _parse_string_time(d[k])
                if dt is not None:
                    return dt.day, dt.month, dt.weekday()

    return 0, 0, 0

def _max_len(dataset):
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
    used = [d for d in (dataset or []) if _get_rating(d) is not None]
    maxLen = max((len(_get_text(d)) for d in used), default=0)
    X, y = [], []
    for d in used:
        X.append(featureQ1(d, maxLen))
        y.append(_get_rating(d))
    if not X:
        return np.array([0.0, 0.0], dtype=float), float("nan")
    X = np.vstack(X).astype(float); y = np.asarray(y, dtype=float)
    theta, *_ = lstsq(X, y, rcond=None)
    mse = float(np.mean((X @ theta - y) ** 2))
    return theta.astype(float), mse

# ---------------- Q2 (19-dim) ----------------
# [1, norm_len] + weekday one-hot (drop Monday) + month one-hot (drop January)
def featureQ2(datum, maxLen):
    s = _get_text(datum)
    norm_len = (len(s) / maxLen) if maxLen > 0 else 0.0
    _, month_num, weekday_num = _get_day_month_weekday(datum)

    # Weekday: keep Tue..Sun (1..6), drop Monday(0) -> 6 dims
    w = np.zeros(6, dtype=float)
    wi = int(weekday_num)
    if 1 <= wi <= 6:
        w[wi - 1] = 1.0

    # Month: keep Feb..Dec (2..12), drop January(1) -> 11 dims
    m = np.zeros(11, dtype=float)
    mi = int(month_num)
    if 2 <= mi <= 12:
        m[mi - 2] = 1.0

    return np.concatenate([[1.0, norm_len], w, m]).astype(float)

def Q2(dataset):
    maxLen_all = _max_len(dataset)  # normalize by global max
    used = [d for d in (dataset or []) if _get_rating(d) is not None]
    X, Y = [], []
    for d in used:
        X.append(featureQ2(d, maxLen_all)); Y.append(_get_rating(d))
    X2 = np.vstack(X) if X else np.zeros((0, 19), dtype=float)
    Y2 = np.asarray(Y, dtype=float)
    if Y2.size == 0:
        return X2, Y2, float("nan")
    theta2, *_ = lstsq(X2, Y2, rcond=None)
    MSE2 = float(np.mean((X2 @ theta2 - Y2) ** 2))
    return X2, Y2, MSE2

# ---------------- Q3 (4-dim) ----------------
# [1, norm_len, weekday_number (0..6), month_number (1..12)]
def featureQ3(datum, maxLen):
    s = _get_text(datum)
    norm_len = (len(s) / maxLen) if maxLen > 0 else 0.0
    _, month_num, weekday_num = _get_day_month_weekday(datum)
    return np.array([1.0, float(norm_len), float(int(weekday_num)), float(int(month_num))], dtype=float)

def Q3(dataset):
    maxLen_all = _max_len(dataset)  # match Q2’s base
    used = [d for d in (dataset or []) if _get_rating(d) is not None]
    X, Y = [], []
    for d in used:
        X.append(featureQ3(d, maxLen_all)); Y.append(_get_rating(d))
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
    maxLen_all = _max_len(dataset)
    X2_all = np.vstack([featureQ2(d, maxLen_all) for d in data])
    X3_all = np.vstack([featureQ3(d, maxLen_all) for d in data])
    Y_all  = np.array([_get_rating(d) for d in data], dtype=float)

    n = len(Y_all); cut = int(0.8 * n)  # runner shuffles upstream
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
    """
    Return precision@[1, 10, 100, 1000] as a list (in this order),
    ranked by predict_proba(class=1) descending.
    """
    Ks = [1, 100, 1000, 10000]
    Xrows, yrows = [], []
    for d in (dataset or []):
        lab = _label_ge4(d)
        if lab is None:
            continue
        Xrows.append(featureQ5(d)); yrows.append(lab)
    if not Xrows:
        return [float('nan')] * len(Ks)

    X = np.vstack(Xrows).astype(float)
    y = np.asarray(yrows, dtype=int)

    clf = LogisticRegression(max_iter=2000, class_weight='balanced', fit_intercept=False)
    clf.fit(X, y)
    scores = clf.predict_proba(X)[:, 1]
    order = np.argsort(-scores)
    y_sorted = y[order]

    out = []
    for K in Ks:
        k = min(K, len(y_sorted))
        out.append(float('nan') if k == 0 else float(np.mean(y_sorted[:k])))
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
