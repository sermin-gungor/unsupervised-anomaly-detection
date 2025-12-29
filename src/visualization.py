import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_pca(X_scaled, result_df, save_path):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    normal = result_df["anomaly"] == 1
    anomaly = result_df["anomaly"] == -1

    plt.figure(figsize=(7,5))
    plt.scatter(X_2d[normal,0], X_2d[normal,1], label="Normal", alpha=0.6)
    plt.scatter(X_2d[anomaly,0], X_2d[anomaly,1], label="Anomali", color="red")
    plt.legend()
    plt.title("One-Class SVM – PCA Görselleştirme")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.savefig(save_path)
    plt.close()

def plot_score_hist(result_df, save_path):
    plt.figure(figsize=(7,5))
    plt.hist(result_df["anomaly_score"], bins=40)
    plt.title("Anomali Skor Dağılımı")
    plt.xlabel("Anomali Skoru")
    plt.ylabel("Frekans")

    plt.savefig(save_path)
    plt.close()
