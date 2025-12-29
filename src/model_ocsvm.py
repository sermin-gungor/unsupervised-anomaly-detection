import pandas as pd
from sklearn.svm import OneClassSVM

def train_ocsvm(X_scaled, nu=0.05, gamma="scale"):
    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    model.fit(X_scaled)
    return model

def detect_anomalies(model, X_scaled, df):
    preds = model.predict(X_scaled)   # 1 = normal, -1 = anomali
    scores = model.decision_function(X_scaled)

    result = df.copy()
    result["anomaly"] = preds
    result["anomaly_score"] = -scores  # büyük skor = daha anormal

    return result
