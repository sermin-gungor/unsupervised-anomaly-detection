import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(csv_path):
    # =========================
    # 1️ Veri setini oku
    # =========================
    df = pd.read_csv(csv_path)

    # =========================
    # 2️ Sayısal kolonları seç
    # (One-Class SVM mesafe tabanlı)
    # =========================
    X = df.select_dtypes(include=["int64", "float64"])

    # =========================
    # 3️ Anlamsız kolonları çıkar (varsa)
    # Örn: ID, index benzeri kolonlar
    # =========================
    drop_cols = [col for col in X.columns if "id" in col.lower()]
    X = X.drop(columns=drop_cols)

    # =========================
    # 4️ Eksik değerleri median ile doldur
    # (outlier'a dayanıklı)
    # =========================
    X = X.fillna(X.median())

    # =========================
    # 5️ Ölçekleme (Z-score)
    # (SVM için KRİTİK)
    # =========================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =========================
    # 6️ Geri dönüş
    # =========================
    return X_scaled, df, X.columns.tolist()
