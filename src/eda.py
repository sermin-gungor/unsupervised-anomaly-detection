import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(csv_path, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    num_df = df.select_dtypes(include=["int64", "float64"])

    # 1️⃣ Histogram + KDE
    for col in num_df.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(num_df[col], kde=True)
        plt.title(f"{col} Dağılımı")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/{col}_hist.png")
        plt.close()

    # 2️⃣ Korelasyon Heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(num_df.corr(), cmap="coolwarm")
    plt.title("Özellikler Arası Korelasyon")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/correlation_heatmap.png")
    plt.close()


def plot_boxplots(df, fig_dir):
    num_df = df.select_dtypes(include=["int64", "float64"])

    for col in num_df.columns:
        plt.figure(figsize=(5,4))
        sns.boxplot(y=num_df[col])
        plt.title(f"{col} Boxplot")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/{col}_boxplot.png")
        plt.close()


def plot_scatter_pairs(df, fig_dir):
    num_df = df.select_dtypes(include=["int64", "float64"])

    if num_df.shape[1] >= 2:
        x, y = num_df.columns[:2]
        plt.figure(figsize=(6,5))
        sns.scatterplot(x=num_df[x], y=num_df[y], alpha=0.6)
        plt.title(f"{x} vs {y}")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/{x}_{y}_scatter.png")
        plt.close()
