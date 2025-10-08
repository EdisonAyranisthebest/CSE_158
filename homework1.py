# ==================== homework1.py (single cell) ====================
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from dateutil import parser as _dateparser

# ------------------ Helpers ------------------
def _get_rating(d):
    return d.get('rating', d.get('overall', d.get('stars')))

def getMaxLen(dataset):
    """Max string length among review text fields for the given iterable of dicts."""
    max_len = 0
    for d in (dataset or []):
        s = d.get('reviewText') or d.get('review_text') or d.get('text') or ''
        max_len = max(max_len, len(str(s)))
    return max_len

# ------------------ Q1 ------------------
def featureQ1(datum, maxLen):
    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')
    norm_len = (len(s) / maxLen) if maxLen else 0.0
    return np.array([1.0, norm_len], dtype=float)

def Q1(dataset):
    """
    Linear regression of rating on [1, normalized_length].
    IMPORTANT: compute maxLen over exactly the rows used for training (those with ratings).
    Return (theta, MSE).
    """
    rows = []
    for d in (dataset or []):
        r = _get_rating(d)
        if r is not None:
            rows.append(d)
    if not rows:
        return np.zeros(2), float('nan')

    maxLen = getMaxLen(rows)
    X = np.vstack([featureQ1(d, maxLen) for d in rows])
    y = np.array([float(_get_rating(d)) for d in rows], dtype=float)

    lr = LinearRegression(fit_intercept=False).fit(X, y)
    theta = lr.coef_
    mse = float(np.mean((X @ theta - y) ** 2))
    return theta, mse

# ------------------ Q2 ------------------
def featureQ2(datum, maxLen):
    """
    Features: [1, norm_len] + weekday one-hot (Mon..Sat, drop Sun) + month one-hot (Jan..Nov, drop Dec)
    Total length = 1 + 1 + 6 + 11 = 19
    """
    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')
    norm_len = (len(s) / maxLen) if maxLen else 0.0

    dt = datum.get('parsed_date')
    if dt is None:
        raw = (datum.get('date_added') or datum.get('reviewTime') or
               datum.get('review_time') or datum.get('date') or datum.get('review_date'))
        if raw:
            try:
                dt = _dateparser.parse(str(raw))
            except Exception:
                dt = None

    w_onehot = [0.0] * 6   # Mon..Sat (drop Sunday)
    m_onehot = [0.0] * 11  # Jan..Nov (drop December)
    if dt is not None:
        w = dt.weekday()   # 0..6
        m = dt.month       # 1..12
        if 0 <= w <= 5:
            w_onehot[w] = 1.0
        if 1 <= m <= 11:
            m_onehot[m - 1] = 1.0

    return np.array([1.0, float(norm_len)] + w_onehot + m_onehot, dtype=float)

def Q2(dataset):
    """
    Return (X2, Y2, MSE2). Autograder uses X2 and Y2; we also compute MSE2.
    IMPORTANT: compute maxLen over exactly the rows used for training (those with ratings).
    """
    rows = []
    for d in (dataset or []):
        r = _get_rating(d)
        if r is not None:
            rows.append(d)

    if not rows:
        return np.zeros((0, 19)), np.array([], dtype=float), float('nan')

    maxLen = getMaxLen(rows)
    X2 = np.vstack([featureQ2(d, maxLen) for d in rows])
    Y2 = np.array([float(_get_rating(d)) for d in rows], dtype=float)

    lr = LinearRegression(fit_intercept=False).fit(X2, Y2)
    MSE2 = float(np.mean((X2 @ lr.coef_ - Y2) ** 2))
    return X2, Y2, MSE2

# ------------------ Q3 ------------------
def featureQ3(datum, maxLen):
    """
    Features: [1, norm_len, weekday_as_number, month_as_number], length = 4.
    """
    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')
    norm_len = (len(s) / maxLen) if maxLen else 0.0

    dt = datum.get('parsed_date')
    if dt is None:
        raw = (datum.get('date_added') or datum.get('reviewTime') or
               datum.get('review_time') or datum.get('date') or datum.get('review_date'))
        if raw:
            try:
                dt = _dateparser.parse(str(raw))
            except Exception:
                dt = None

    if dt is None:
        w, m = 0.0, 1.0
    else:
        w, m = float(dt.weekday()), float(dt.month)

    return np.array([1.0, float(norm_len), w, m], dtype=float)

def Q3(dataset):
    """
    Return (X3, Y3, MSE3).
    IMPORTANT: compute maxLen over exactly the rows used for training (those with ratings).
    """
    rows = []
    for d in (dataset or []):
        r = _get_rating(d)
        if r is not None:
            rows.append(d)

    if not rows:
        return np.zeros((0, 4)), np.array([], dtype=float), float('nan')

    maxLen = getMaxLen(rows)
    X3 = np.vstack([featureQ3(d, maxLen) for d in rows])
    Y3 = np.array([float(_get_rating(d)) for d in rows], dtype=float)

    lr = LinearRegression(fit_intercept=False).fit(X3, Y3)
    MSE3 = float(np.mean((X3 @ lr.coef_ - Y3) ** 2))
    return X3, Y3, MSE3

# ------------------ Q4 ------------------
def Q4(dataset):
    """
    Split first half for train, second half for test (runner already shuffles if needed).
    Train Q2 and Q3 encodings on train; report test MSEs.
    """
    n = len(dataset)
    train, test = dataset[: n // 2], dataset[n // 2 :]
    maxLen_tr = getMaxLen(train)

    # Q2 model
    X2_tr = np.vstack([featureQ2(d, maxLen_tr) for d in train]) if train else np.zeros((0, 19))
    y_tr = np.array([float(_get_rating(d)) for d in train], dtype=float)
    lr2 = LinearRegression(fit_intercept=False)
    if len(y_tr):
        lr2.fit(X2_tr, y_tr)

    # Q3 model
    X3_tr = np.vstack([featureQ3(d, maxLen_tr) for d in train]) if train else np.zeros((0, 4))
    lr3 = LinearRegression(fit_intercept=False)
    if len(y_tr):
        lr3.fit(X3_tr, y_tr)

    # Test
    y_te = np.array([float(_get_rating(d)) for d in test], dtype=float)
    X2_te = np.vstack([featureQ2(d, maxLen_tr) for d in test]) if test else np.zeros((0, 19))
    X3_te = np.vstack([featureQ3(d, maxLen_tr) for d in test]) if test else np.zeros((0, 4))

    pred2 = X2_te @ getattr(lr2, 'coef_', np.zeros(19))
    pred3 = X3_te @ getattr(lr3, 'coef_', np.zeros(4))
    test_mse2 = float(np.mean((pred2 - y_te) ** 2)) if len(y_te) else float('nan')
    test_mse3 = float(np.mean((pred3 - y_te) ** 2)) if len(y_te) else float('nan')
    return test_mse2, test_mse3

# ------------------ Q5 / Q6 / Q7 ------------------
def featureQ5(datum):
    """
    Baseline features (NO explicit bias term): [review_length, exclamation_count]
    """
    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')
    return np.array([float(len(s)), float(s.count('!'))], dtype=float)

def _label_from_review_overall(d):
    """
    Label = 1{ review/overall >= 4 }.
    Accepts flat 'review/overall' or nested d['review']['overall'].
    Returns 0/1 or None if missing.
    """
    if 'review/overall' in d:
        rating = d['review/overall']
    elif isinstance(d.get('review'), dict) and 'overall' in d['review']:
        rating = d['review']['overall']
    else:
        return None
    try:
        return 1 if float(rating) >= 4.0 else 0
    except Exception:
        return None

def Q5(dataset, feat_func):
    """
    Train LogisticRegression on given dataset with feat_func.
    Return (TP, TN, FP, FN, BER).
    NOTE: Use deterministic config without class_weight to match grader.
    """
    X, y = [], []
    for d in (dataset or []):
        lab = _label_from_review_overall(d)
        if lab is None:
            continue
        y.append(lab)
        X.append(feat_func(d))
    if not X:
        return 0, 0, 0, 0, float('nan')

    X = np.vstack(X)
    y = np.array(y, dtype=int)

    clf = LogisticRegression(solver='liblinear', max_iter=1000, random_state=0)
    clf.fit(X, y)
    yp = clf.predict(X)

    TP = int(((yp == 1) & (y == 1)).sum())
    TN = int(((yp == 0) & (y == 0)).sum())
    FP = int(((yp == 1) & (y == 0)).sum())
    FN = int(((yp == 0) & (y == 1)).sum())

    P = max(int((y == 1).sum()), 1)
    N = max(int((y == 0).sum()), 1)
    BER = 0.5 * ((FN / P) + (FP / N))
    return TP, TN, FP, FN, float(BER)

def Q6(dataset):
    """
    Precision@K for K in {1, 10, 100, 1000} using featureQ5 and
    LogisticRegression(class_weight='balanced') ONLY.
    Rank by decision_function with stable sorting to match grader behavior.
    """
    X, y = [], []
    for d in (dataset or []):
        lab = _label_from_review_overall(d)
        if lab is None:
            continue
        y.append(lab)
        X.append(featureQ5(d))
    if not X:
        return [0.0, 0.0, 0.0, 0.0]

    X = np.vstack(X)
    y = np.array(y, dtype=int)

    clf = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000, random_state=0)
    clf.fit(X, y)

    if hasattr(clf, 'decision_function'):
        scores = clf.decision_function(X)
    else:
        scores = clf.predict_proba(X)[:, 1]

    # Stable sort so ties are handled deterministically
    order = np.argsort(-scores, kind='mergesort')
    y_sorted = y[order]

    Ks = [1, 10, 100, 1000]
    precs = []
    for K in Ks:
        k = min(K, len(y_sorted))
        precs.append(0.0 if k == 0 else float(y_sorted[:k].sum()) / k)
    return precs

def featureQ7(datum):
    """
    Stronger features for Q7 (NO explicit bias term), same classifier rule as Q5.
    """
    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')
    toks = [t.strip(".,!?;:()[]{}'\"").lower() for t in s.split() if t]

    pos = {
        "good","great","excellent","amazing","love","loved","awesome",
        "fantastic","perfect","best","wonderful","favorite","happy",
        "tasty","fresh","crisp","smooth","aroma","balanced","complex"
    }
    neg = {
        "bad","terrible","awful","hate","hated","worst","poor",
        "disappointing","boring","broken","flat","stale","skunky",
        "bitter","sour","thin","watery","metallic","off"
    }

    pos_cnt = float(sum(t in pos for t in toks))
    neg_cnt = float(sum(t in neg for t in toks))
    bal     = pos_cnt - neg_cnt
    length  = float(len(s))
    emarks  = float(s.count('!'))
    qmarks  = float(s.count('?'))
    digits  = float(sum(ch.isdigit() for ch in s))
    caps_ratio = sum(1 for ch in s if ch.isalpha() and ch.isupper()) / (1.0 + len(s))
    avg_wlen = (sum(len(t) for t in toks) / len(toks)) if toks else 0.0

    return np.array([
        length, emarks, qmarks,
        pos_cnt, neg_cnt, bal,
        avg_wlen, digits, float(caps_ratio)
    ], dtype=float)

def Q7(dataset):
    """
    Compare BER for baseline features (featureQ5) vs improved features (featureQ7).
    Return (BER5, BER7).
    """
    _, _, _, _, BER5 = Q5(dataset, featureQ5)
    _, _, _, _, BER7 = Q5(dataset, featureQ7)
    return BER5, BER7
# ==================== end of file ====================
