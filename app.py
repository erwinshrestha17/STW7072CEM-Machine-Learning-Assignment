import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from math import pi
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as ss

# ------------------------------
# CONFIGURATION
# ------------------------------
sns.set(style="whitegrid")
plt.switch_backend("Agg")

OUTPUT_DIR = "output_plots"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=40, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()


def predict_with_progress(model, X, batch_size=1000, task_name="Testing"):
    n_samples = X.shape[0]
    predictions = []
    probabilities = []
    n_batches = int(np.ceil(n_samples / batch_size))

    print(f"\nStarting {task_name}...")

    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_samples)
        X_batch = X.iloc[start:end]

        batch_pred = model.predict(X_batch)
        try:
            if hasattr(model, "predict_proba"):
                batch_proba = model.predict_proba(X_batch)[:, 1]
            else:
                batch_proba = np.zeros(len(batch_pred))
        except:
            batch_proba = np.zeros(len(batch_pred))

        predictions.extend(batch_pred)
        probabilities.extend(batch_proba)
        print_progress_bar(i + 1, n_batches, prefix=f'{task_name}', suffix='Complete', length=30)

    return np.array(predictions), np.array(probabilities)


# ------------------------------
# PHASE 1 — LOAD + CLEAN DATA
# ------------------------------
def load_and_clean_data(file_path):
    print(f"\n--- Loading Dataset from {file_path} ---")
    df = pd.read_csv(file_path)

    fill_vals = {
        "trekking_route": "None",
        "visa_type": "Unknown",
        "hotel_name": "Unknown"
    }

    for col, val in fill_vals.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # Handle numerics
    df[df.select_dtypes(include="number").columns] = df.select_dtypes(include="number").fillna(0)

    # Handle categoricals
    df[df.select_dtypes(include="object").columns] = df.select_dtypes(include="object").fillna("Unknown")

    # Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    print(f"Removed {before - after} duplicate rows.")

    # Create spending features
    if "adr" in df.columns:
        df["daily_spending_usd"] = df["adr"]
        if "stays_in_weekend_nights" in df.columns and "stays_in_week_nights" in df.columns:
            df["total_nights"] = df["stays_in_week_nights"] + df["stays_in_weekend_nights"]
            df["total_spending_usd"] = df["adr"] * df["total_nights"]

    return df


# ------------------------------
# PHASE 2 — FULL EDA + FE DIAGRAMS
# ------------------------------
def perform_eda(df):

    print("\n--- Generating EDA Diagrams ---")

    # 1 – Missing Value Heatmap
    plt.figure(figsize=(12,6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Value Heatmap")
    plt.savefig(os.path.join(OUTPUT_DIR, "1_missing_values.png"))
    plt.close()

    # 2 – Numerical Distributions
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    for col in num_cols:
        plt.figure(figsize=(8,5))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"dist_{col}.png"))
        plt.close()

    # 3 – Outlier Boxplots
    for col in num_cols:
        plt.figure(figsize=(8,5))
        sns.boxplot(x=df[col])
        plt.title(f"Outlier Check: {col}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"box_{col}.png"))
        plt.close()

    # 4 – Categorical Frequency Plots
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        plt.figure(figsize=(10,5))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f"Frequency of {col}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"cat_{col}.png"))
        plt.close()

    # 5 – Full Correlation Heatmap
    plt.figure(figsize=(14,10))
    sns.heatmap(df.corr(), cmap="coolwarm")
    plt.title("Full Correlation Heatmap")
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_full.png"))
    plt.close()

    # 6 – VIF (Multicollinearity)
    X_vif = df.select_dtypes(include=['float64','int64']).dropna()
    vif_df = pd.DataFrame()
    vif_df["Feature"] = X_vif.columns
    vif_df["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
    vif_df.to_csv(os.path.join(OUTPUT_DIR, "vif_scores.csv"), index=False)

    # 7 – Cancellation vs Hotel / Market / Country
    if "is_canceled" in df.columns:
        for col in ["hotel", "market_segment", "country"]:
            if col in df.columns:
                plt.figure(figsize=(10,6))
                sns.barplot(x=df[col], y=df["is_canceled"])
                plt.xticks(rotation=45)
                plt.title(f"Cancellation Rate by {col}")
                plt.savefig(os.path.join(OUTPUT_DIR, f"cancel_{col}.png"))
                plt.close()

    # 8 – Lead Time vs Cancellation
    if "lead_time" in df.columns and "is_canceled" in df.columns:
        plt.figure(figsize=(8,5))
        sns.scatterplot(x=df["lead_time"], y=df["is_canceled"], alpha=0.3)
        plt.title("Lead Time vs Cancellation")
        plt.savefig(os.path.join(OUTPUT_DIR, "leadtime_cancel.png"))
        plt.close()

    # 9 – Cramér's V for categorical-target relationship
    def cramers_v(x, y):
        confusion = pd.crosstab(x, y)
        chi2 = ss.chi2_contingency(confusion)[0]
        n = confusion.sum().sum()
        return np.sqrt(chi2 / (n * (min(confusion.shape)-1)))

    if "is_canceled" in df.columns:
        cramers = {}
        for col in cat_cols:
            cramers[col] = cramers_v(df[col], df["is_canceled"])
        pd.Series(cramers).sort_values(ascending=False).to_csv(
            os.path.join(OUTPUT_DIR, "cramers_v.csv")
        )

    # 10 – Spending Tier Plot
    if "daily_spending_usd" in df.columns:
        df["Spending_Tier"] = pd.qcut(
            df["daily_spending_usd"], 3, labels=["Low", "Medium", "High"]
        )
        plt.figure(figsize=(10,5))
        sns.countplot(x="Spending_Tier", hue="is_canceled", data=df)
        plt.title("Cancellation by Spending Tier")
        plt.savefig(os.path.join(OUTPUT_DIR, "spending_tier.png"))
        plt.close()

    # 11 – Pairplot
    core_cols = ["lead_time","daily_spending_usd","total_spending_usd","adults","children"]
    core_cols = [c for c in core_cols if c in df.columns]

    if len(core_cols) >= 2:
        sns.pairplot(df[core_cols], diag_kind="kde")
        plt.savefig(os.path.join(OUTPUT_DIR, "pairplot_core.png"))
        plt.close()

    print("EDA Completed Successfully.")
    return df


# ------------------------------
# PHASE 3 — CLUSTERING + ELBOW
# ------------------------------
def perform_clustering(df):

    features = ["daily_spending_usd","total_spending_usd","lead_time","stays_in_week_nights","adults"]
    features = [f for f in features if f in df.columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["Cluster_ID"] = kmeans.fit_predict(X_scaled)

    # Cluster Scatter Plot
    plt.figure(figsize=(10,6))
    sns.scatterplot(
        x=df["daily_spending_usd"],
        y=df["total_spending_usd"],
        hue=df["Cluster_ID"],
        palette="viridis"
    )
    plt.title("K-Means Socio-Economic Clusters")
    plt.savefig(os.path.join(OUTPUT_DIR, "clusters.png"))
    plt.close()

    # Elbow Plot
    inertias = []
    K_range = range(1,10)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8,5))
    plt.plot(K_range, inertias, "bo-")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.savefig(os.path.join(OUTPUT_DIR, "elbow_plot.png"))
    plt.close()

    # Radar Chart
    cluster_summary = df.groupby("Cluster_ID")[features].mean()

    categories = cluster_summary.columns
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(8,8))
    for idx in cluster_summary.index:
        values = cluster_summary.loc[idx].tolist()
        values += values[:1]
        plt.polar(angles, values, marker='o', label=f"Cluster {idx}")

    plt.title("Cluster Profiles (Radar)")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_radar.png"))
    plt.close()

    return df


# ------------------------------
# PHASE 4 — MODEL TRAINING & EVALUATION
# ------------------------------
def perform_prediction(df):

    df_encoded = df.copy()
    cat_cols = df_encoded.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

    if "is_canceled" not in df.columns:
        print("Target variable missing!")
        return

    X = df_encoded.drop("is_canceled", axis=1)
    y = df_encoded["is_canceled"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Logistic Regression
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)
    y_pred_lr, y_prob_lr = predict_with_progress(
        lr, X_test, task_name="LR Testing"
    )

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf, y_prob_rf = predict_with_progress(
        rf, X_test, task_name="RF Testing"
    )

    # Accuracy
    print(f"\nLR Accuracy: {accuracy_score(y_test, y_pred_lr):.2%}")
    print(f"RF Accuracy: {accuracy_score(y_test, y_pred_rf):.2%}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Random Forest)")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    # ROC CURVES
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

    plt.figure(figsize=(7,5))
    plt.plot(fpr_lr, tpr_lr, label=f"LR (AUC={auc(fpr_lr,tpr_lr):.2f})")
    plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC={auc(fpr_rf,tpr_rf):.2f})")
    plt.plot([0,1],[0,1],"k--")
    plt.legend()
    plt.title("ROC Curves")
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
    plt.close()

    # Feature Importance
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    feat_imp[:10].plot(kind="barh")
    plt.title("Top 10 Features Driving Cancellation")
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.close()


# ------------------------------
# MAIN EXECUTION
# ------------------------------
if __name__ == "__main__":
    df = load_and_clean_data("hotel_bookings_new.csv")
    df = perform_eda(df)
    df = perform_clustering(df)
    perform_prediction(df)

    print("\nAll diagrams generated successfully!")
