import numpy as np
from sklearn.linear_model import LinearRegression
from dateutil import parser as _dateparser
from collections import defaultdict
from sklearn import linear_model
import numpy
import math
from sklearn.linear_model import LogisticRegression

# ---------- Q1 ----------
def getMaxLen(dataset):
    maxLen = 0
    for d in dataset or []:
        txt = d.get('reviewText') or d.get('review_text') or d.get('text') or ''
        maxLen = max(maxLen, len(str(txt)))
    return maxLen

def featureQ1(datum, maxLen):
    txt = datum.get('reviewText') or datum.get('review_text') or datum.get('text') or ''
    L = len(str(txt))
    norm_len = (L / maxLen) if maxLen else 0.0
    return np.array([1.0, norm_len], dtype=float)

def Q1(dataset):
    maxLen = getMaxLen(dataset)
    X, y = [], []
    for d in dataset or []:
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
    MSE = float(np.mean((X @ theta - y) ** 2))
    return theta, MSE

# ---------- Q2 ----------
def featureQ2(datum, maxLen):
    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')
    norm_len = (len(s) / maxLen) if maxLen else 0.0
    dt = datum.get('parsed_date')
    if dt is None:
        raw = (datum.get('date_added') or datum.get('reviewTime') or
               datum.get('review_time') or datum.get('date') or datum.get('review_date'))
        if raw:
            try: dt = _dateparser.parse(str(raw))
            except Exception: dt = None
    w_onehot = [0.0]*6   # Mon..Sat; drop Sun
    m_onehot = [0.0]*11  # Jan..Nov; drop Dec
    if dt is not None:
        w = dt.weekday()
        m = dt.month
        if 0 <= w <= 5: w_onehot[w] = 1.0
        if 1 <= m <= 11: m_onehot[m-1] = 1.0
    return np.array([1.0, float(norm_len)] + w_onehot + m_onehot, dtype=float)

def Q2(dataset):
    maxLen = getMaxLen(dataset)
    X, Y = [], []
    for d in dataset or []:
        y = d.get('rating', d.get('overall', d.get('stars')))
        if y is None: continue
        X.append(featureQ2(d, maxLen)); Y.append(float(y))
    X2 = np.vstack(X) if X else np.zeros((0,19))
    Y2 = np.array(Y, dtype=float)
    if len(Y2) == 0: return X2, Y2, float('nan')
    lr = LinearRegression(fit_intercept=False).fit(X2, Y2)
    MSE2 = float(np.mean((X2 @ lr.coef_ - Y2) ** 2))
    return X2, Y2, MSE2

# ---------- Q3 ----------
def featureQ3(datum, maxLen):
    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')
    norm_len = (len(s) / maxLen) if maxLen else 0.0
    dt = datum.get('parsed_date')
    if dt is None:
        raw = (datum.get('date_added') or datum.get('reviewTime') or
               datum.get('review_time') or datum.get('date') or datum.get('review_date'))
        if raw:
            try: dt = _dateparser.parse(str(raw))
            except Exception: dt = None
    if dt is None: w, m = 0, 1
    else: w, m = dt.weekday(), dt.month
    return np.array([1.0, float(norm_len), float(w), float(m)], dtype=float)

def Q3(dataset):
    maxLen = getMaxLen(dataset)
    X, Y = [], []
    for d in dataset or []:
        y = d.get('rating', d.get('overall', d.get('stars')))
        if y is None: continue
        X.append(featureQ3(d, maxLen)); Y.append(float(y))
    X3 = np.vstack(X) if X else np.zeros((0,4))
    Y3 = np.array(Y, dtype=float)
    if len(Y3) == 0: return X3, Y3, float('nan')
    lr = LinearRegression(fit_intercept=False).fit(X3, Y3)
    MSE3 = float(np.mean((X3 @ lr.coef_ - Y3) ** 2))
    return X3, Y3, MSE3

# ---------- Q4 ----------
def Q4(dataset):
    n = len(dataset); split = n // 2
    train, test = dataset[:split], dataset[split:]
    maxLen_tr = getMaxLen(train)
    # Q2 model
    X2_tr = np.vstack([featureQ2(d, maxLen_tr) for d in train]) if train else np.zeros((0,19))
    y_tr  = np.array([float(d.get('rating', d.get('overall', d.get('stars')))) for d in train], dtype=float)
    lr2 = LinearRegression(fit_intercept=False); 
    if len(y_tr): lr2.fit(X2_tr, y_tr)
    # Q3 model
    X3_tr = np.vstack([featureQ3(d, maxLen_tr) for d in train]) if train else np.zeros((0,4))
    lr3 = LinearRegression(fit_intercept=False);
    if len(y_tr): lr3.fit(X3_tr, y_tr)
    # test
    y_te  = np.array([float(d.get('rating', d.get('overall', d.get('stars')))) for d in test], dtype=float)
    X2_te = np.vstack([featureQ2(d, maxLen_tr) for d in test]) if test else np.zeros((0,19))
    X3_te = np.vstack([featureQ3(d, maxLen_tr) for d in test]) if test else np.zeros((0,4))
    pred2 = X2_te @ getattr(lr2, 'coef_', np.zeros(19))
    pred3 = X3_te @ getattr(lr3, 'coef_', np.zeros(4))
    test_mse2 = float(np.mean((pred2 - y_te) ** 2)) if len(y_te) else float('nan')
    test_mse3 = float(np.mean((pred3 - y_te) ** 2)) if len(y_te) else float('nan')
    return test_mse2, test_mse3

# ---------- Q5 ----------
def featureQ5(datum):
    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')
    return np.array([1.0, float(len(s)), float(s.count('!'))], dtype=float)

def Q5(dataset, feat_func):
    X, y = [], []
    for d in dataset or []:
        # Try many possible label keys
        lab = (
            d.get('label', d.get('y', d.get('target',
            d.get('sentiment', d.get('polarity', d.get('class',
            d.get('truth', d.get('liked', d.get('recommended',
            d.get('is_positive'))))))))))
        )

        if lab is None:
            rating = d.get('rating', d.get('overall', d.get('stars',
                     d.get('review_overall', d.get('review/overall', d.get('beer/overall'))))))
            )
            if rating is not None:
                try:
                    lab = 1 if float(rating) >= 4.0 else 0
                except Exception:
                    lab = None

        if lab is None:
            continue

        if isinstance(lab, str):
            s = lab.strip().lower()
            if s in ('1','pos','positive','true','yes','y','t','recommended','recommends'):
                lab = 1
            elif s in ('0','neg','negative','false','no','n','f','not_recommended','not recommended'):
                lab = 0
            else:
                try:
                    lab = 1 if float(s) >= 4.0 else 0
                except Exception:
                    continue
        elif isinstance(lab, bool):
            lab = int(lab)
        elif isinstance(lab, (int, float)):
            
            if lab in (-1, 1):
                lab = 1 if lab > 0 else 0
            elif lab in (0, 1):
                lab = int(lab)
            else:
                lab = 1 if float(lab) >= 4.0 else 0
        else:
            continue

        y.append(int(lab))
        X.append(feat_func(d))

    if not X:
        return 0,0,0,0,float('nan')

    X = np.vstack(X); y = np.array(y, dtype=int)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    yp = clf.predict(X)

    TP = int(((yp==1)&(y==1)).sum())
    TN = int(((yp==0)&(y==0)).sum())
    FP = int(((yp==1)&(y==0)).sum())
    FN = int(((yp==0)&(y==1)).sum())
    P  = max(int((y==1).sum()), 1)
    N  = max(int((y==0).sum()), 1)
    BER = 0.5*((FN/P) + (FP/N))
    return TP, TN, FP, FN, float(BER)

# ---------- Q6 ----------
def Q6(dataset):
    X, y = [], []
    for d in dataset or []:
        lab = (
            d.get('label', d.get('y', d.get('target',
            d.get('sentiment', d.get('polarity', d.get('class',
            d.get('truth', d.get('liked', d.get('recommended',
            d.get('is_positive'))))))))))
        )
        if lab is None:
            rating = d.get('rating', d.get('overall', d.get('stars',
                     d.get('review_overall', d.get('review/overall', d.get('beer/overall'))))))
            )
            if rating is not None:
                try:
                    lab = 1 if float(rating) >= 4.0 else 0
                except Exception:
                    lab = None
        if lab is None:
            continue

        if isinstance(lab, str):
            s = lab.strip().lower()
            if s in ('1','pos','positive','true','yes','y','t','recommended','recommends'):
                lab = 1
            elif s in ('0','neg','negative','false','no','n','f','not_recommended','not recommended'):
                lab = 0
            else:
                try:
                    lab = 1 if float(s) >= 4.0 else 0
                except Exception:
                    continue
        elif isinstance(lab, bool):
            lab = int(lab)
        elif isinstance(lab, (int, float)):
            if lab in (-1, 1):
                lab = 1 if lab > 0 else 0
            elif lab in (0, 1):
                lab = int(lab)
            else:
                lab = 1 if float(lab) >= 4.0 else 0
        else:
            continue

        y.append(int(lab))
        X.append(featureQ5(d))

    if not X:
        return []

    X = np.vstack(X); y = np.array(y, dtype=int)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    scores = clf.predict_proba(X)[:,1] if hasattr(clf,'predict_proba') else clf.decision_function(X)

    order = np.argsort(-scores)
    y_sorted = y[order]
    K = min(100, len(y_sorted))
    precs, tp = [], 0
    for k in range(1, K+1):
        if y_sorted[k-1] == 1:
            tp += 1
        precs.append(tp / k)
    return precs

# ---------- Q7 ----------
def featureQ7(datum):
    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')
    toks = [t.strip(".,!?;:()[]{}'\"").lower() for t in s.split() if t]

    pos = {"good","great","excellent","amazing","love","loved","awesome",
           "fantastic","perfect","best","wonderful","favorite","happy",
           "tasty","fresh","crisp","smooth"}
    neg = {"bad","terrible","awful","hate","hated","worst","poor",
           "disappointing","boring","broken","sad","angry",
           "stale","flat","skunky","bitter"}  

    pos_cnt = float(sum(t in pos for t in toks))
    neg_cnt = float(sum(t in neg for t in toks))
    bal     = pos_cnt - neg_cnt
    qmarks  = float(s.count('?'))
    emarks  = float(s.count('!'))
    caps_ratio = sum(1 for ch in s if ch.isalpha() and ch.isupper()) / (1.0 + len(s))
    digits  = float(sum(ch.isdigit() for ch in s))
    length  = float(len(s))

    return np.array([
        1.0, length, emarks, qmarks,
        pos_cnt, neg_cnt, bal,
        digits, float(caps_ratio)
    ], dtype=float)

def Q7(dataset):
    _,_,_,_, BER5 = Q5(dataset, featureQ5)
    _,_,_,_, BER7 = Q5(dataset, featureQ7)
    return BER5, BER7
