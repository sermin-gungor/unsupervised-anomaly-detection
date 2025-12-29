import numpy as np

def anomaly_ratio(result_df):
    """
    Tespit edilen anomali oranı
    """
    ratio = (result_df["anomaly"] == -1).mean()
    return ratio


def nu_sensitivity_test(X_scaled, df, train_func, detect_func, nu_list):
    """
    Farklı nu değerlerinde anomali sayısını test eder
    """
    results = []

    for nu in nu_list:
        model = train_func(X_scaled, nu=nu)
        tmp_df = detect_func(model, X_scaled, df)
        count = (tmp_df["anomaly"] == -1).sum()

        results.append({
            "nu": nu,
            "anomaly_count": count
        })

    return results


def extreme_value_check(result_df, feature_names):
    """
    Anomaliler gerçekten uç değer mi?
    """
    anomalies = result_df[result_df["anomaly"] == -1]
    normals = result_df[result_df["anomaly"] == 1]

    return {
        "anomaly_stats": anomalies[feature_names].describe(),
        "normal_stats": normals[feature_names].describe()
    }
