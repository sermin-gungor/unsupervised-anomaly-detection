from preprocessing import load_and_preprocess
from model_ocsvm import train_ocsvm, detect_anomalies
from visualization import plot_pca, plot_score_hist
from eda import run_eda, plot_boxplots, plot_scatter_pairs
from validation import anomaly_ratio, nu_sensitivity_test

import os

DATA_PATH = "data/Students Social Media Addiction.csv"
FIG_DIR = "results/figures"
EDA_DIR = f"{FIG_DIR}/eda"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

# =========================
# 1️ EDA (HAM VERİ)
# =========================
run_eda(
    csv_path=DATA_PATH,
    fig_dir=EDA_DIR
)

# =========================
# 2️ PREPROCESSING
# =========================
X_scaled, df, feature_names = load_and_preprocess(DATA_PATH)

# EDA'nın df isteyen grafikleri
plot_boxplots(df, EDA_DIR)
plot_scatter_pairs(df, EDA_DIR)

# =========================
# 3️ MODEL
# =========================
model = train_ocsvm(X_scaled, nu=0.05)

# =========================
# 4️ ANOMALİ TESPİTİ
# =========================
result_df = detect_anomalies(model, X_scaled, df)

# =========================
# 5️ KAYIT
# =========================
result_df.to_csv("results/anomalies.csv", index=False)

# =========================
# 6️ MODEL GRAFİKLERİ
# =========================
plot_pca(X_scaled, result_df, f"{FIG_DIR}/pca.png")
plot_score_hist(result_df, f"{FIG_DIR}/score_hist.png")

# =========================
# 7️ ÖZET
# =========================
n_anomaly = (result_df["anomaly"] == -1).sum()
with open("results/summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Toplam kayıt sayısı: {len(result_df)}\n")
    f.write(f"Tespit edilen anomali sayısı: {n_anomaly}\n")

print(" Proje başarıyla çalıştı.")
# =========================
# 8️ MODEL VALIDATION
# =========================
ratio = anomaly_ratio(result_df)
print(f"Anomali oranı: {ratio:.3f}")

nu_results = nu_sensitivity_test(
    X_scaled,
    df,
    train_ocsvm,
    detect_anomalies,
    nu_list=[0.01, 0.03, 0.05, 0.1]
)

with open("results/summary.txt", "a", encoding="utf-8") as f:
    f.write("\n--- Model Validation ---\n")
    f.write(f"Anomali Oranı: {ratio:.3f}\n")
    f.write("nu Duyarlılık Testi:\n")
    for r in nu_results:
        f.write(f"  nu={r['nu']} -> anomali sayısı={r['anomaly_count']}\n")
