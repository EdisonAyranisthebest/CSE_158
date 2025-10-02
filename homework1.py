#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
from sklearn.linear_model import LinearRegression
from dateutil import parser as _dateparser
from collections import defaultdict
from sklearn import linear_model
import numpy
import math
from sklearn.linear_model import LogisticRegression


# In[39]:


### Question 1


# In[40]:


def getMaxLen(dataset):
    maxLen = 0
    for d in dataset or []:
        txt = d.get('reviewText') or d.get('review_text') or d.get('text') or ''
        maxLen = max(maxLen, len(str(txt)))
    return maxLen


# In[41]:


def featureQ1(datum, maxLen):
    txt = datum.get('reviewText') or datum.get('review_text') or datum.get('text') or ''
    L = len(str(txt))
    norm_len = (L / maxLen) if maxLen else 0.0
    return np.array([1.0, norm_len], dtype=float)


# In[42]:


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

    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, y)

    theta = lr.coef_
    preds = X @ theta
    MSE = float(np.mean((preds - y) ** 2))
    return theta, MSE


# In[43]:


### Question 2


# In[44]:


def featureQ2(datum, maxLen):
    txt = datum.get('reviewText') or datum.get('review_text') or datum.get('text') or ''
    L = len(str(txt))
    x_len = (L / maxLen) if maxLen else 0.0

    raw = datum.get('reviewTime') or datum.get('review_time') or datum.get('date') or datum.get('review_date')
    day, month = 1, 1
    if raw:
        try:
            dt = _dateparser.parse(str(raw))
            day, month = dt.day, dt.month
        except Exception:
            pass

    return np.array([1.0, float(x_len), float(day), float(month)], dtype=float)


# In[45]:


def Q2(dataset):
    maxLen = getMaxLen(dataset)
    X, Y = [], []
    for d in dataset or []:
        y = d.get('rating', d.get('overall', d.get('stars')))
        if y is None:
            continue
        X.append(featureQ2(d, maxLen))
        Y.append(float(y))

    X2 = np.vstack(X) if len(X) else np.zeros((0, 4))
    Y2 = np.array(Y, dtype=float)

    if len(Y2) == 0:
        return X2, Y2, float('nan')

    lr = LinearRegression(fit_intercept=False)
    lr.fit(X2, Y2)
    preds = X2 @ lr.coef_
    MSE2 = float(np.mean((preds - Y2) ** 2))
    return X2, Y2, MSE2


# In[46]:


### Question 3


# In[47]:


def featureQ3(datum, maxLen):
    base = featureQ2(datum, maxLen)  

    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')
    words = s.split()
    word_count   = float(len(words))
    exclam_count = float(s.count('!'))
    caps_ratio   = sum(1 for ch in s if ch.isalpha() and ch.isupper()) / (1.0 + len(s))

    extra = np.array([word_count, exclam_count, float(caps_ratio)], dtype=float)
    return np.concatenate([base, extra])


# In[48]:


def Q3(dataset):
    maxLen = getMaxLen(dataset)
    X, Y = [], []
    for d in dataset or []:
        y = d.get('rating', d.get('overall', d.get('stars')))
        if y is None:
            continue
        X.append(featureQ3(d, maxLen))
        Y.append(float(y))

    X3 = np.vstack(X) if len(X) else np.zeros((0, 7))
    Y3 = np.array(Y, dtype=float)

    if len(Y3) == 0:
        return X3, Y3, float('nan')

    lr = LinearRegression(fit_intercept=False)
    lr.fit(X3, Y3)
    preds = X3 @ lr.coef_
    MSE3 = float(np.mean((preds - Y3) ** 2))
    return X3, Y3, MSE3


# In[49]:


### Question 4


# In[58]:


def Q4(dataset):
    n = len(dataset)
    split = n // 2
    train, test = dataset[:split], dataset[split:]

    # Q2 features (baseline)
    maxLen_train = getMaxLen(train)
    X2_train = np.vstack([featureQ2(d, maxLen_train) for d in train])
    y2_train = np.array([float(d['rating']) for d in train])
    theta2 = np.linalg.lstsq(X2_train, y2_train, rcond=None)[0]

    X2_test = np.vstack([featureQ2(d, maxLen_train) for d in test])
    y2_test = np.array([float(d['rating']) for d in test])
    mse2 = np.mean((X2_test @ theta2 - y2_test) ** 2)

    # Q3 features (richer)
    maxLen_train = getMaxLen(train)
    X3_train = np.vstack([featureQ3(d, maxLen_train) for d in train])
    y3_train = np.array([float(d['rating']) for d in train])
    theta3 = np.linalg.lstsq(X3_train, y3_train, rcond=None)[0]

    X3_test = np.vstack([featureQ3(d, maxLen_train) for d in test])
    y3_test = np.array([float(d['rating']) for d in test])
    mse3 = np.mean((X3_test @ theta3 - y3_test) ** 2)

    return mse2, mse3


# In[51]:


### Question 5


# In[52]:


def featureQ5(datum):
    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')
    return np.array([1.0, float(len(s)), float(s.count('!'))], dtype=float)


# In[53]:


def Q5(dataset, feat_func):
    X, y = [], []
    for d in dataset or []:
        lab = d.get('label', d.get('y', d.get('target')))
        if lab is None:
            continue
        if isinstance(lab, str):
            lab = 1 if lab.lower() in ('1', 'pos', 'positive', 'true', 'yes') else 0
        y.append(int(lab))
        X.append(feat_func(d))

    if not X:
        return 0, 0, 0, 0, float('nan')

    X = np.vstack(X)
    y = np.array(y, dtype=int)

    # Balanced so the model doesn't predict everything as 1
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
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


# In[54]:


### Question 6


# In[55]:


def Q6(dataset):
    X, y = [], []
    for d in dataset or []:
        lab = d.get('label', d.get('y', d.get('target')))
        if lab is None:
            continue
        if isinstance(lab, str):
            lab = 1 if lab.lower() in ('1', 'pos', 'positive', 'true', 'yes') else 0
        y.append(int(lab))
        X.append(featureQ5(d))

    if not X:
        return []

    X = np.vstack(X)
    y = np.array(y, dtype=int)

    clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
    clf.fit(X, y)
    scores = clf.predict_proba(X)[:, 1] 

    order = np.argsort(-scores)
    y_sorted = y[order]
    K = min(100, len(y_sorted))

    precs, tp = [], 0
    for k in range(1, K + 1):
        if y_sorted[k - 1] == 1:
            tp += 1
        precs.append(tp / k)
    return precs


# In[56]:


### Question 7


# In[57]:


def featureQ7(datum):
    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')

    # simple lexicons (tiny but effective)
    pos_words = {"good","great","excellent","amazing","love","loved","awesome",
                 "fantastic","perfect","best","wonderful","favorite","happy"}
    neg_words = {"bad","terrible","awful","hate","hated","worst","poor",
                 "disappointing","disappointed","boring","broken","sad","angry"}

    tokens = [t.strip(".,!?;:()[]{}'\"").lower() for t in s.split() if t]
    pos_cnt = sum(t in pos_words for t in tokens)
    neg_cnt = sum(t in neg_words for t in tokens)

    # other stylistic signals
    qmarks = s.count('?')
    emarks = s.count('!')
    caps_ratio = sum(1 for ch in s if ch.isalpha() and ch.isupper()) / (1.0 + len(s))
    digits = sum(ch.isdigit() for ch in s)

    return np.array([
        1.0,
        float(len(s)),           
        float(emarks),           
        float(qmarks),            
        float(pos_cnt),          
        float(neg_cnt),           
        float(pos_cnt - neg_cnt),
        float(digits),            
        float(caps_ratio)         
    ], dtype=float)


def Q7(dataset):
    """
    Compare BER with baseline (featureQ5) vs improved (featureQ7).
    Returns (BER5, BER7).
    """
    _, _, _, _, BER5 = Q5(dataset, featureQ5)
    _, _, _, _, BER7 = Q5(dataset, featureQ7)
    return BER5, BER7


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




