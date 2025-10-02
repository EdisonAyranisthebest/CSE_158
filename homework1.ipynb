{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "07ad6b17",
   "metadata": {
    "id": "07ad6b17"
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from dateutil import parser as _dateparser\n",
    "from collections import defaultdict\n",
    "from sklearn import linear_model\n",
    "import numpy\n",
    "import math\n",
    "from sklearn.linear_model import LogisticRegression\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "ab71afc7-f078-4439-ba7d-5e348f44266c",
   "metadata": {
    "id": "ab71afc7-f078-4439-ba7d-5e348f44266c"
   },
   "outputs": [],
   "source": [
    "### Question 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "f164c2d8-828a-4af2-ba69-000fad8f1c54",
   "metadata": {
    "id": "f164c2d8-828a-4af2-ba69-000fad8f1c54"
   },
   "outputs": [],
   "source": [
    "def getMaxLen(dataset):\n",
    "    maxLen = 0\n",
    "    for d in dataset or []:\n",
    "        txt = d.get('reviewText') or d.get('review_text') or d.get('text') or ''\n",
    "        maxLen = max(maxLen, len(str(txt)))\n",
    "    return maxLen"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "9dc58396-c9f4-4a94-b62e-74e03064ca24",
   "metadata": {
    "id": "9dc58396-c9f4-4a94-b62e-74e03064ca24"
   },
   "outputs": [],
   "source": [
    "def featureQ1(datum, maxLen):\n",
    "    txt = datum.get('reviewText') or datum.get('review_text') or datum.get('text') or ''\n",
    "    L = len(str(txt))\n",
    "    norm_len = (L / maxLen) if maxLen else 0.0\n",
    "    return np.array([1.0, norm_len], dtype=float)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "f1ebbbb2-d5ae-405e-a02d-7307265387b8",
   "metadata": {
    "id": "f1ebbbb2-d5ae-405e-a02d-7307265387b8"
   },
   "outputs": [],
   "source": [
    "def Q1(dataset):\n",
    "    maxLen = getMaxLen(dataset)\n",
    "    X, y = [], []\n",
    "    for d in dataset or []:\n",
    "        rating = d.get('rating', d.get('overall', d.get('stars')))\n",
    "        if rating is None:\n",
    "            continue\n",
    "        X.append(featureQ1(d, maxLen))\n",
    "        y.append(float(rating))\n",
    "\n",
    "    if not X:\n",
    "        return np.zeros(2), float('nan')\n",
    "\n",
    "    X = np.vstack(X)\n",
    "    y = np.array(y, dtype=float)\n",
    "\n",
    "    lr = LinearRegression(fit_intercept=False)\n",
    "    lr.fit(X, y)\n",
    "\n",
    "    theta = lr.coef_\n",
    "    preds = X @ theta\n",
    "    MSE = float(np.mean((preds - y) ** 2))\n",
    "    return theta, MSE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "a8e3b548-53bc-47ad-bc45-d63fc4af95ce",
   "metadata": {
    "id": "a8e3b548-53bc-47ad-bc45-d63fc4af95ce"
   },
   "outputs": [],
   "source": [
    "### Question 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "abce94cf-a7ca-4eef-b5fa-920349ee1003",
   "metadata": {
    "id": "abce94cf-a7ca-4eef-b5fa-920349ee1003"
   },
   "outputs": [],
   "source": [
    "def featureQ2(datum, maxLen):\n",
    "    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')\n",
    "    norm_len = (len(s) / maxLen) if maxLen else 0.0\n",
    "\n",
    "    dt = datum.get('parsed_date')\n",
    "    if dt is None:\n",
    "        raw = (datum.get('date_added') or datum.get('reviewTime') or\n",
    "               datum.get('review_time') or datum.get('date') or datum.get('review_date'))\n",
    "        if raw:\n",
    "            try:\n",
    "                dt = _dateparser.parse(str(raw))\n",
    "            except Exception:\n",
    "                dt = None\n",
    "\n",
    "    w_onehot = [0.0]*6   \n",
    "    m_onehot = [0.0]*11  \n",
    "    if dt is not None:\n",
    "        w = dt.weekday()        \n",
    "        m = dt.month           \n",
    "        if 0 <= w <= 5: w_onehot[w] = 1.0\n",
    "        if 1 <= m <= 11: m_onehot[m-1] = 1.0\n",
    "\n",
    "    return np.array([1.0, float(norm_len)] + w_onehot + m_onehot, dtype=float)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "8cb8f5cf",
   "metadata": {
    "id": "8cb8f5cf"
   },
   "outputs": [],
   "source": [
    "def Q2(dataset):\n",
    "    maxLen = getMaxLen(dataset)\n",
    "    X, Y = [], []\n",
    "    for d in dataset or []:\n",
    "        y = d.get('rating', d.get('overall', d.get('stars')))\n",
    "        if y is None: \n",
    "            continue\n",
    "        X.append(featureQ2(d, maxLen))\n",
    "        Y.append(float(y))\n",
    "    X2 = np.vstack(X) if X else np.zeros((0, 19))\n",
    "    Y2 = np.array(Y, dtype=float)\n",
    "    if len(Y2) == 0:\n",
    "        return X2, Y2, float('nan')\n",
    "    lr = LinearRegression(fit_intercept=False)\n",
    "    lr.fit(X2, Y2)\n",
    "    MSE2 = float(np.mean((X2 @ lr.coef_ - Y2)**2))\n",
    "    return X2, Y2, MSE2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "35d18540-6413-4459-8082-0f8cc1d3e276",
   "metadata": {
    "id": "35d18540-6413-4459-8082-0f8cc1d3e276"
   },
   "outputs": [],
   "source": [
    "### Question 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "939c02d3-eb26-4e65-96a2-d55fc41d2209",
   "metadata": {
    "id": "939c02d3-eb26-4e65-96a2-d55fc41d2209"
   },
   "outputs": [],
   "source": [
    "def featureQ3(datum, maxLen):\n",
    "    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')\n",
    "    norm_len = (len(s) / maxLen) if maxLen else 0.0\n",
    "\n",
    "    dt = datum.get('parsed_date')\n",
    "    if dt is None:\n",
    "        raw = (datum.get('date_added') or datum.get('reviewTime') or\n",
    "               datum.get('review_time') or datum.get('date') or datum.get('review_date'))\n",
    "        if raw:\n",
    "            try:\n",
    "                dt = _dateparser.parse(str(raw))\n",
    "            except Exception:\n",
    "                dt = None\n",
    "\n",
    "    if dt is None:\n",
    "        w, m = 0, 1\n",
    "    else:\n",
    "        w, m = dt.weekday(), dt.month\n",
    "\n",
    "    return np.array([1.0, float(norm_len), float(w), float(m)], dtype=float)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "3efaacd2-6ac8-4a3c-8d32-60ad8f389917",
   "metadata": {
    "id": "3efaacd2-6ac8-4a3c-8d32-60ad8f389917"
   },
   "outputs": [],
   "source": [
    "def Q3(dataset):\n",
    "    maxLen = getMaxLen(dataset)\n",
    "    X, Y = [], []\n",
    "    for d in dataset or []:\n",
    "        y = d.get('rating', d.get('overall', d.get('stars')))\n",
    "        if y is None: \n",
    "            continue\n",
    "        X.append(featureQ3(d, maxLen))\n",
    "        Y.append(float(y))\n",
    "    X3 = np.vstack(X) if X else np.zeros((0, 4))\n",
    "    Y3 = np.array(Y, dtype=float)\n",
    "    if len(Y3) == 0:\n",
    "        return X3, Y3, float('nan')\n",
    "    lr = LinearRegression(fit_intercept=False)\n",
    "    lr.fit(X3, Y3)\n",
    "    MSE3 = float(np.mean((X3 @ lr.coef_ - Y3)**2))\n",
    "    return X3, Y3, MSE3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "6d89e772-b1ad-46fc-aeb8-45a6df15fa5d",
   "metadata": {
    "id": "6d89e772-b1ad-46fc-aeb8-45a6df15fa5d"
   },
   "outputs": [],
   "source": [
    "### Question 4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "d1029574-b4c4-4675-83ad-c20674a2428b",
   "metadata": {
    "id": "d1029574-b4c4-4675-83ad-c20674a2428b"
   },
   "outputs": [],
   "source": [
    "def Q4(dataset):\n",
    "    n = len(dataset)\n",
    "    split = n // 2\n",
    "    train, test = dataset[:split], dataset[split:]\n",
    "\n",
    "    maxLen_tr = getMaxLen(train)\n",
    "\n",
    "    # Train Q2\n",
    "    X2_tr = np.vstack([featureQ2(d, maxLen_tr) for d in train]) if train else np.zeros((0,19))\n",
    "    y_tr  = np.array([float(d.get('rating', d.get('overall', d.get('stars')))) for d in train], dtype=float)\n",
    "    lr2 = LinearRegression(fit_intercept=False)\n",
    "    if len(y_tr): lr2.fit(X2_tr, y_tr)\n",
    "\n",
    "    # Train Q3\n",
    "    X3_tr = np.vstack([featureQ3(d, maxLen_tr) for d in train]) if train else np.zeros((0,4))\n",
    "    lr3 = LinearRegression(fit_intercept=False)\n",
    "    if len(y_tr): lr3.fit(X3_tr, y_tr)\n",
    "\n",
    "    # Test using train normalization\n",
    "    y_te  = np.array([float(d.get('rating', d.get('overall', d.get('stars')))) for d in test], dtype=float)\n",
    "    X2_te = np.vstack([featureQ2(d, maxLen_tr) for d in test]) if test else np.zeros((0,19))\n",
    "    X3_te = np.vstack([featureQ3(d, maxLen_tr) for d in test]) if test else np.zeros((0,4))\n",
    "\n",
    "    pred2 = X2_te @ getattr(lr2, 'coef_', np.zeros(19))\n",
    "    pred3 = X3_te @ getattr(lr3, 'coef_', np.zeros(4))\n",
    "    test_mse2 = float(np.mean((pred2 - y_te)**2)) if len(y_te) else float('nan')\n",
    "    test_mse3 = float(np.mean((pred3 - y_te)**2)) if len(y_te) else float('nan')\n",
    "    return test_mse2, test_mse3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "3a38cc78-c502-4ad1-b2db-d8358c0caa3e",
   "metadata": {
    "id": "3a38cc78-c502-4ad1-b2db-d8358c0caa3e"
   },
   "outputs": [],
   "source": [
    "### Question 5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "4df2e6ea",
   "metadata": {
    "id": "4df2e6ea"
   },
   "outputs": [],
   "source": [
    "def featureQ5(datum):\n",
    "    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')\n",
    "    return np.array([1.0, float(len(s)), float(s.count('!'))], dtype=float)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "cabb75ec-826a-497e-b158-f5c340aea4d2",
   "metadata": {
    "id": "cabb75ec-826a-497e-b158-f5c340aea4d2"
   },
   "outputs": [],
   "source": [
    "def Q5(dataset, feat_func):\n",
    "    X, y = [], []\n",
    "    for d in dataset or []:\n",
    "        lab = d.get('label', d.get('y', d.get('target')))\n",
    "        if lab is None:\n",
    "            continue\n",
    "        if isinstance(lab, str):\n",
    "            lab = 1 if lab.lower() in ('1','pos','positive','true','yes') else 0\n",
    "        y.append(int(lab))\n",
    "        X.append(feat_func(d))\n",
    "    if not X:\n",
    "        return 0,0,0,0,float('nan')\n",
    "\n",
    "    X = np.vstack(X); y = np.array(y, dtype=int)\n",
    "    clf = LogisticRegression(max_iter=1000)\n",
    "    clf.fit(X, y)\n",
    "    yp = clf.predict(X)\n",
    "\n",
    "    TP = int(((yp==1)&(y==1)).sum())\n",
    "    TN = int(((yp==0)&(y==0)).sum())\n",
    "    FP = int(((yp==1)&(y==0)).sum())\n",
    "    FN = int(((yp==0)&(y==1)).sum())\n",
    "    P  = max(int((y==1).sum()), 1)\n",
    "    N  = max(int((y==0).sum()), 1)\n",
    "    BER = 0.5*((FN/P) + (FP/N))\n",
    "    return TP, TN, FP, FN, float(BER)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "a3c14955-d174-473b-bf2b-a2b98776685e",
   "metadata": {
    "id": "a3c14955-d174-473b-bf2b-a2b98776685e"
   },
   "outputs": [],
   "source": [
    "### Question 6"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "2a71178c-589e-433c-8bdb-70c5e1510fce",
   "metadata": {
    "id": "2a71178c-589e-433c-8bdb-70c5e1510fce"
   },
   "outputs": [],
   "source": [
    "def Q6(dataset):\n",
    "    X, y = [], []\n",
    "    for d in dataset or []:\n",
    "        lab = d.get('label', d.get('y', d.get('target')))\n",
    "        if lab is None:\n",
    "            continue\n",
    "        if isinstance(lab, str):\n",
    "            lab = 1 if lab.lower() in ('1','pos','positive','true','yes') else 0\n",
    "        y.append(int(lab))\n",
    "        X.append(featureQ5(d))\n",
    "    if not X:\n",
    "        return []\n",
    "\n",
    "    X = np.vstack(X); y = np.array(y, dtype=int)\n",
    "    clf = LogisticRegression(max_iter=1000)\n",
    "    clf.fit(X, y)\n",
    "    scores = clf.predict_proba(X)[:,1] if hasattr(clf,'predict_proba') else clf.decision_function(X)\n",
    "\n",
    "    order = np.argsort(-scores)\n",
    "    y_sorted = y[order]\n",
    "    K = min(100, len(y_sorted))\n",
    "    precs, tp = [], 0\n",
    "    for k in range(1, K+1):\n",
    "        if y_sorted[k-1] == 1:\n",
    "            tp += 1\n",
    "        precs.append(tp / k)\n",
    "    return precs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "e8291f55-cfa1-4979-bcaa-42a238e9a844",
   "metadata": {
    "id": "e8291f55-cfa1-4979-bcaa-42a238e9a844"
   },
   "outputs": [],
   "source": [
    "### Question 7"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "96e16d5a",
   "metadata": {
    "id": "96e16d5a"
   },
   "outputs": [],
   "source": [
    "def featureQ7(datum):\n",
    "    s = str(datum.get('reviewText') or datum.get('review_text') or datum.get('text') or '')\n",
    "    toks = [t.strip(\".,!?;:()[]{}'\\\"\").lower() for t in s.split() if t]\n",
    "\n",
    "    pos = {\"good\",\"great\",\"excellent\",\"amazing\",\"love\",\"loved\",\"awesome\",\n",
    "           \"fantastic\",\"perfect\",\"best\",\"wonderful\",\"favorite\",\"happy\"}\n",
    "    neg = {\"bad\",\"terrible\",\"awful\",\"hate\",\"hated\",\"worst\",\"poor\",\n",
    "           \"disappointing\",\"boring\",\"broken\",\"sad\",\"angry\"}\n",
    "\n",
    "    pos_cnt = float(sum(t in pos for t in toks))\n",
    "    neg_cnt = float(sum(t in neg for t in toks))\n",
    "    bal     = pos_cnt - neg_cnt\n",
    "    caps_ratio = sum(1 for ch in s if ch.isalpha() and ch.isupper()) / (1.0 + len(s))\n",
    "    digits = float(sum(ch.isdigit() for ch in s))\n",
    "\n",
    "    return np.array([1.0, float(len(s)), float(s.count('!')), float(s.count('?')),\n",
    "                     pos_cnt, neg_cnt, bal, digits, float(caps_ratio)], dtype=float)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "024a628f",
   "metadata": {
    "id": "024a628f"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c6e53829-2a24-4657-b9ac-0b8a888dec1f",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ccba735d-0aec-46b4-880d-fafb72d1aea7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cacc5595-f50b-49a9-9afb-d1cf53247830",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "01da2482-2977-48fc-bfed-52583558d595",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d02db7fe-45bf-4a47-849d-e9c42c11e53e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d2628eaa-aa17-486a-8566-f81086b0d457",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9e1150ac-5fc8-4a50-8a76-a2bbaab39d46",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7d5d400b-5ec4-416b-814a-79c46fa60a98",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7fdd8f66-070f-471b-925f-efa66f56db24",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
