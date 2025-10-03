import numpy as np
from sklearn.linear_model import LinearRegression
from dateutil import parser as _dateparser
from collections import defaultdict
from sklearn import linear_model
import numpy
import math
from sklearn.linear_model import LogisticRegression

# ------------------ Q1 ------------------
def getMaxLen(dataset):
    max_len = 0
    for d in (dataset or []):
        s = d.get('reviewText') or d.get('review_text') or d.get('text') or ''
        max_len = max(max_len, len(str(s)))
    return max_len

def featureQ1(datum, maxLen):
    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')
    norm_len = (len(s) / maxLen) if maxLen else 0.0
    return np.array([1.0, norm_len], dtype=float)

def Q1(dataset):
    """
    Return (theta, MSE) for a linear regression of rating on [1, normalized_length].
    """
    maxLen = getMaxLen(dataset)
    X, y = [], []
    for d in (dataset or []):
        rating = d.get('rating', d.get('overall', d.get('stars')))
        if rating is None:
            continue
        X.append(featureQ1(d, maxLen))
        y.append(float(rating))
    if not X:
        return np.zeros(2), float('nan')
    X = np.vstack(X)
    y = np.array(y, dtype=float)
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
    """
    maxLen = getMaxLen(dataset)
    X, Y = [], []
    for d in (dataset or []):
        y = d.get('rating', d.get('overall', d.get('stars')))
        if y is None:
            continue
        X.append(featureQ2(d, maxLen))
        Y.append(float(y))
    X2 = np.vstack(X) if X else np.zeros((0, 19))
    Y2 = np.array(Y, dtype=float)
    if len(Y2) == 0:
        return X2, Y2, float('nan')
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
    """
    maxLen = getMaxLen(dataset)
    X, Y = [], []
    for d in (dataset or []):
        y = d.get('rating', d.get('overall', d.get('stars')))
        if y is None:
            continue
        X.append(featureQ3(d, maxLen))
        Y.append(float(y))
    X3 = np.vstack(X) if X else np.zeros((0, 4))
    Y3 = np.array(Y, dtype=float)
    if len(Y3) == 0:
        return X3, Y3, float('nan')
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
    y_tr = np.array([float(d.get('rating', d.get('overall', d.get('stars')))) for d in train], dtype=float)
    lr2 = LinearRegression(fit_intercept=False)
    if len(y_tr):
        lr2.fit(X2_tr, y_tr)

    # Q3 model
    X3_tr = np.vstack([featureQ3(d, maxLen_tr) for d in train]) if train else np.zeros((0, 4))
    lr3 = LinearRegression(fit_intercept=False)
    if len(y_tr):
        lr3.fit(X3_tr, y_tr)

    # Test
    y_te = np.array([float(d.get('rating', d.get('overall', d.get('stars')))) for d in test], dtype=float)
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
    return np.array([1.0, float(len(s)), float(s.count('!'))], dtype=float)

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
    Train LogisticRegression(class_weight='balanced') on given dataset with feat_func.
    Return (TP, TN, FP, FN, BER). No additional params; no splitting/shuffling here.
    """
    X, y = [], []
    for d in (dataset or []):
        lab = d.get('label', d.get('y', d.get('target')))
        if lab is None:
            continue
        # normalize label to {0,1}
        if isinstance(lab, str):
            s = lab.strip().lower()
            if s in ('1','true','yes','y','t','pos','positive'):
                lab = 1
            elif s in ('0','false','no','n','f','neg','negative'):
                lab = 0
            else:
                # if it's a numeric string, fall back safely
                try:
                    lab = int(float(s) > 0.5)
                except Exception:
                    continue
        elif isinstance(lab, bool):
            lab = int(lab)
        else:
            lab = int(lab)

        y.append(lab)
        X.append(feat_func(d))

    if not X:
        return 0, 0, 0, 0, float('nan')

    X = np.vstack(X)
    y = np.array(y, dtype=int)

    # Balanced logistic regression, no extra params
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X, y)
    yp = clf.predict(X)

    TP = int(((yp == 1) & (y == 1)).sum())
    TN = int(((yp == 0) & (y == 0)).sum())
    FP = int(((yp == 1) & (y == 0)).sum())
    FN = int(((yp == 0) & (y == 1)).sum())
    P  = max(int((y == 1).sum()), 1)
    N  = max(int((y == 0).sum()), 1)
    BER = 0.5 * ((FN / P) + (FP / N))
    return TP, TN, FP, FN, float(BER)

def Q6(dataset):
    """
    Precision@K for K in {1, 10, 100, 1000} using featureQ5 and
    LogisticRegression(class_weight='balanced') ONLY.
    """
    X, y = [], []
    for d in (dataset or []):
        lab = d.get('label', d.get('y', d.get('target')))
        if lab is None:
            continue
        if isinstance(lab, str):
            s = lab.strip().lower()
            if s in ('1','true','yes','y','t','pos','positive'):
                lab = 1
            elif s in ('0','false','no','n','f','neg','negative'):
                lab = 0
            else:
                try:
                    lab = int(float(s) > 0.5)
                except Exception:
                    continue
        elif isinstance(lab, bool):
            lab = int(lab)
        else:
            lab = int(lab)

        y.append(lab)
        X.append(featureQ5(d))

    if not X:
        return [0.0, 0.0, 0.0, 0.0]

    X = np.vstack(X)
    y = np.array(y, dtype=int)

    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X, y)

    scores = clf.predict_proba(X)[:, 1] if hasattr(clf, 'predict_proba') else clf.decision_function(X)
    order = np.argsort(-scores)
    y_sorted = y[order]

    Ks = [1, 10, 100, 1000]
    precs = []
    for K in Ks:
        k = min(K, len(y_sorted))
        precs.append(float(y_sorted[:k].sum()) / k if k > 0 else 0.0)
    return precs
    
def featureQ7(datum):
    """
    Stronger features for Q7 (NO explicit bias term), same classifier rule as Q5.
    """
    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')
    toks = [t.strip(".,!?;:()[]{}'\"").lower() for t in s.split() if t]

    pos = {"good","great","excellent","amazing","love","loved","awesome","fantastic","perfect","best"}
    neg = {"bad","terrible","awful","hate","hated","worst","poor","boring","flat","stale","bitter","sour"}

    pos_cnt = float(sum(t in pos for t in toks))
    neg_cnt = float(sum(t in neg for t in toks))
    bal     = pos_cnt - neg_cnt
    length  = float(len(s))
    emarks  = float(s.count('!'))
    qmarks  = float(s.count('?'))
    digits  = float(sum(ch.isdigit() for ch in s))
    caps_ratio = sum(1 for ch in s if ch.isalpha() and ch.isupper()) / (1.0 + len(s))
    avg_wlen = (sum(len(t) for t in toks) / len(toks)) if toks else 0.0

    return np.array([length, emarks, qmarks, pos_cnt, neg_cnt, bal, avg_wlen, digits, float(caps_ratio)], dtype=float)

def Q7(dataset):
    """
    Compare BER for baseline features (featureQ5) vs improved features (featureQ7).
    """
    _, _, _, _, BER5 = Q5(dataset, featureQ5)
    _, _, _, _, BER7 = Q5(dataset, featureQ7)
    return BER5, BER7
