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
    return max(len(str(d.get('reviewText') or d.get('review_text') or d.get('text') or ''))) for d in dataset)

def featureQ1(datum, maxLen):
    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')
    norm_len = len(s) / maxLen if maxLen > 0 else 0.0
    return np.array([1.0, norm_len], dtype=float)

def Q1(dataset):
    X, y = [], []
    maxLen = getMaxLen(dataset)
    for d in dataset:
        if 'overall' not in d:
            continue
        y.append(float(d['overall']))
        X.append(featureQ1(d, maxLen))
    if not X:
        return np.zeros(2), np.array([]), np.nan
    X = np.vstack(X)
    y = np.array(y)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    preds = X @ theta
    mse = float(((preds - y) ** 2).mean())
    return theta, y, mse

# ------------------ Q2 ------------------
def featureQ2(datum, maxLen):
    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')
    norm_len = len(s) / maxLen if maxLen > 0 else 0.0
    date = datum.get('reviewTime') or datum.get('date')
    try:
        dt = datetime.strptime(date, "%m %d, %Y")
        weekday = dt.weekday()   # 0=Mon,…6=Sun
        month = dt.month
    except Exception:
        weekday, month = 0, 1
    onehot_weekday = [1.0 if i == weekday else 0.0 for i in range(6)]  # drop Sunday
    onehot_month = [1.0 if i == month else 0.0 for i in range(11)]     # drop December
    return np.array([1.0, norm_len] + onehot_weekday + onehot_month, dtype=float)

def Q2(dataset):
    X, y = [], []
    maxLen = getMaxLen(dataset)
    for d in dataset:
        if 'overall' not in d:
            continue
        y.append(float(d['overall']))
        X.append(featureQ2(d, maxLen))
    if not X:
        return np.zeros(19), np.array([]), np.nan
    X = np.vstack(X)
    y = np.array(y)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    preds = X @ theta
    mse = float(((preds - y) ** 2).mean())
    return theta, y, mse

# ------------------ Q3 ------------------
def featureQ3(datum, maxLen):
    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')
    norm_len = len(s) / maxLen if maxLen > 0 else 0.0
    date = datum.get('reviewTime') or datum.get('date')
    try:
        dt = datetime.strptime(date, "%m %d, %Y")
        weekday = float(dt.weekday())
        month = float(dt.month)
    except Exception:
        weekday, month = 0.0, 1.0
    return np.array([1.0, norm_len, weekday, month], dtype=float)

def Q3(dataset):
    X, y = [], []
    maxLen = getMaxLen(dataset)
    for d in dataset:
        if 'overall' not in d:
            continue
        y.append(float(d['overall']))
        X.append(featureQ3(d, maxLen))
    if not X:
        return np.zeros(4), np.array([]), np.nan
    X = np.vstack(X)
    y = np.array(y)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    preds = X @ theta
    mse = float(((preds - y) ** 2).mean())
    return theta, y, mse

# ------------------ Q4 ------------------
def Q4(dataset):
    # dataset is a list already shuffled by runner
    n = len(dataset)
    train, test = dataset[:n//2], dataset[n//2:]

    # Train on Q2 features
    maxLen_train = getMaxLen(train)
    X2_train, y2_train = [], []
    for d in train:
        if 'overall' not in d: continue
        y2_train.append(float(d['overall']))
        X2_train.append(featureQ2(d, maxLen_train))
    X2_train = np.vstack(X2_train); y2_train = np.array(y2_train)
    theta2, *_ = np.linalg.lstsq(X2_train, y2_train, rcond=None)

    # Test Q2
    X2_test, y2_test = [], []
    for d in test:
        if 'overall' not in d: continue
        y2_test.append(float(d['overall']))
        X2_test.append(featureQ2(d, maxLen_train))
    X2_test = np.vstack(X2_test); y2_test = np.array(y2_test)
    mse2 = float(((X2_test @ theta2 - y2_test) ** 2).mean())

    # Train on Q3 features
    X3_train, y3_train = [], []
    for d in train:
        if 'overall' not in d: continue
        y3_train.append(float(d['overall']))
        X3_train.append(featureQ3(d, maxLen_train))
    X3_train = np.vstack(X3_train); y3_train = np.array(y3_train)
    theta3, *_ = np.linalg.lstsq(X3_train, y3_train, rcond=None)

    # Test Q3
    X3_test, y3_test = [], []
    for d in test:
        if 'overall' not in d: continue
        y3_test.append(float(d['overall']))
        X3_test.append(featureQ3(d, maxLen_train))
    X3_test = np.vstack(X3_test); y3_test = np.array(y3_test)
    mse3 = float(((X3_test @ theta3 - y3_test) ** 2).mean())

    return mse2, mse3

# ------------------ Q5 ------------------
def featureQ5(datum):
    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')
    return np.array([float(len(s)), float(s.count('!'))], dtype=float)

def _label_from_review_overall(d):
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
    X, y = [], []
    for d in dataset:
        lab = _label_from_review_overall(d)
        if lab is None: continue
        y.append(lab)
        X.append(feat_func(d))
    if not X: return 0, 0, 0, 0, float('nan')
    X = np.vstack(X); y = np.array(y)
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X, y)
    yp = clf.predict(X)
    TP = int(((yp==1)&(y==1)).sum())
    TN = int(((yp==0)&(y==0)).sum())
    FP = int(((yp==1)&(y==0)).sum())
    FN = int(((yp==0)&(y==1)).sum())
    P  = max(int((y==1).sum()),1)
    N  = max(int((y==0).sum()),1)
    BER = 0.5*((FN/P)+(FP/N))
    return TP, TN, FP, FN, float(BER)

# ------------------ Q6 ------------------
def Q6(dataset):
    X, y = [], []
    for d in dataset:
        lab = _label_from_review_overall(d)
        if lab is None: continue
        y.append(lab)
        X.append(featureQ5(d))
    if not X: return [0.0,0.0,0.0,0.0]
    X = np.vstack(X); y = np.array(y)
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X, y)
    scores = clf.predict_proba(X)[:,1]
    order = np.argsort(-scores)
    y_sorted = y[order]
    Ks = [1,10,100,1000]
    precs = []
    for K in Ks:
        k = min(K, len(y_sorted))
        precs.append(float(y_sorted[:k].sum())/k if k>0 else 0.0)
    return precs

# ------------------ Q7 ------------------
def featureQ7(datum):
    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')
    toks = [t.strip(".,!?;:()[]{}'\"").lower() for t in s.split() if t]
    pos = {"good","great","excellent","amazing","love","awesome","fantastic","perfect"}
    neg = {"bad","terrible","awful","hate","worst","poor","boring","flat","stale"}
    pos_cnt = float(sum(t in pos for t in toks))
    neg_cnt = float(sum(t in neg for t in toks))
    bal = pos_cnt - neg_cnt
    length = float(len(s))
    emarks = float(s.count('!'))
    qmarks = float(s.count('?'))
    digits = float(sum(ch.isdigit() for ch in s))
    caps_ratio = sum(1 for ch in s if ch.isalpha() and ch.isupper())/(1.0+len(s))
    avg_wlen = (sum(len(t) for t in toks)/len(toks)) if toks else 0.0
    return np.array([length, emarks, qmarks, pos_cnt, neg_cnt, bal, avg_wlen, digits, float(caps_ratio)], dtype=float)

def Q7(dataset):
    _,_,_,_,BER5 = Q5(dataset, featureQ5)
    _,_,_,_,BER7 = Q5(dataset, featureQ7)
    return BER5, BER7


def Q7(dataset):
    _, _, _, _, BER5 = Q5(dataset, featureQ5)
    _, _, _, _, BER7 = Q5(dataset, featureQ7)
    return BER5, BER7
