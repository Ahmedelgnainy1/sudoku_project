from sklearn.linear_model import LogisticRegression

def train_logreg(x, y, seed=42, max_iter=200):
    m = LogisticRegression(max_iter=max_iter, random_state=seed)
    m.fit(x.reshape(len(x), -1), y - 1)
    return m

def logreg_proba(m, x):
    return m.predict_proba(x.reshape(len(x), -1))