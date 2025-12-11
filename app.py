import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from math import pi
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, roc_curve, auc,
    confusion_matrix, log_loss, mean_squared_error,
    mean_absolute_error, silhouette_score,
    precision_recall_curve, average_precision_score
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ------------------------------
# 1. CONFIGURATION
# ------------------------------
# Use 'Agg' backend to save plots to file without a display window (prevents crashes)
plt.switch_backend("Agg")
sns.set(style="whitegrid")

OUTPUT_DIR = "output_plots"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"--- SYSTEM: Saving all outputs to '{OUTPUT_DIR}' ---")


# ------------------------------
# 2. HELPER FUNCTIONS
# ------------------------------
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=40, fill='█'):
    """Creates a terminal progress bar."""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()


def predict_with_progress(model, X, batch_size=2000, task_name="Testing"):
    """Runs prediction in batches to show progress on large datasets."""
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


def plot_learning_curve(estimator, X, y, title, filename):
    """Generates a learning curve to check for overfitting/underfitting."""
    print(f"  >> Generating Learning Curve for {title} (this may take a moment)...")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=3, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="accuracy"
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.title(f"Learning Curve: {title}")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()


# ------------------------------
# 3. PHASE 1: LOAD & CLEAN
# ------------------------------
def load_and_clean_data(file_path):
    print(f"\n--- [Phase 1] Loading Dataset from {file_path} ---")
    if not os.path.exists(file_path):
        print(f"ERROR: File '{file_path}' not found.")
        return None

    df = pd.read_csv(file_path)

    # 1. Fill Nulls with business logic
    fill_vals = {"trekking_route": "None", "visa_type": "Unknown", "hotel_name": "Unknown"}
    for col, val in fill_vals.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    df[df.select_dtypes(include="number").columns] = df.select_dtypes(include="number").fillna(0)
    df[df.select_dtypes(include="object").columns] = df.select_dtypes(include="object").fillna("Unknown")

    # 2. Remove Duplicates
    initial_len = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  >> Removed {initial_len - len(df)} duplicate rows.")

    # 3. Feature Engineering (Spending Metrics)
    if "adr" in df.columns:
        df["daily_spending_usd"] = df["adr"]
        if "stays_in_weekend_nights" in df.columns and "stays_in_week_nights" in df.columns:
            df["total_nights"] = df["stays_in_week_nights"] + df["stays_in_weekend_nights"]
            # Avoid zero division or multiplication issues
            df["total_spending_usd"] = df["adr"] * df["total_nights"]

    return df


# ------------------------------
# 4. PHASE 2: TARGETED EDA
# ------------------------------
def perform_eda(df):
    print("\n--- [Phase 2] Generating Essential EDA Plots ---")

    # 1. Target Distribution (Is the data balanced?)
    if "is_canceled" in df.columns:
        plt.figure(figsize=(6, 5))
        sns.countplot(x=df["is_canceled"], hue=df["is_canceled"], palette='viridis', legend=False)
        plt.title("Distribution of Cancellations (Target)")
        plt.xlabel("Is Canceled (0=No, 1=Yes)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "1_target_distribution.png"))
        plt.close()

    # 2. Key Feature Distributions (Histograms)
    # Only plotting the most critical financial/temporal features for the report
    key_features = ['lead_time', 'daily_spending_usd', 'total_spending_usd', 'age']
    for feat in key_features:
        if feat in df.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[feat], kde=True, color='skyblue')
            plt.title(f"Distribution of {feat}")
            plt.xlabel(feat)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"dist_{feat}.png"))
            plt.close()

    # 3. Correlation Heatmap (Multicollinearity Check)
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False, linewidths=0.5)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "2_correlation_heatmap.png"))
        plt.close()

    # 4. Missing Value Heatmap (Data Quality Check)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Data Completeness Check")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_data_quality.png"))
    plt.close()

    return df


# ------------------------------
# 5. PHASE 3: CLUSTERING & PROFILING
# ------------------------------
def perform_clustering(df):
    print("\n--- [Phase 3] K-Means Clustering & Validation ---")

    # Select features that define "Socio-Economic Profile"
    features = ["daily_spending_usd", "total_spending_usd", "lead_time", "stays_in_week_nights", "adults"]
    features = [f for f in features if f in df.columns]

    if not features:
        print("  >> Clustering features missing, skipping Phase 3.")
        return df

    # Standardize Data (Critical for K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # 1. Elbow Method
    inertias = []
    K_range = range(1, 10)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertias, "bo-", linewidth=2)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method (Determining k)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_elbow_method.png"))
    plt.close()

    # 2. Apply K-Means (k=4 based on previous analysis)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["Cluster_ID"] = kmeans.fit_predict(X_scaled)

    # 3. Silhouette Score (VALIDATION)
    # Using sample_size to speed up execution on large datasets
    sil_score = silhouette_score(X_scaled, df["Cluster_ID"], sample_size=10000, random_state=42)
    print(f"  >> Clustering Validation (Silhouette Score): {sil_score:.4f} (Close to 1 is best)")

    # 4. Cluster Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=df["daily_spending_usd"],
        y=df["total_spending_usd"],
        hue=df["Cluster_ID"],
        palette="viridis", s=60, alpha=0.7
    )
    plt.title("Socio-Economic Clusters (Spending Behavior)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_cluster_scatter.png"))
    plt.close()

    # 5. Cluster Boxplot (Interpretation)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster_ID', y='daily_spending_usd', data=df, hue='Cluster_ID', palette="viridis", legend=False)
    plt.title("Spending Profile by Cluster (Profiling Interpretation)")
    plt.ylim(0, 500)  # Limit y-axis to focus on main distribution
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "6_cluster_profile_boxplot.png"))
    plt.close()

    return df


# ------------------------------
# 6. PHASE 4: PREDICTION & METRICS
# ------------------------------
def calculate_metrics(y_true, y_prob, model_name, n_features=0, n_samples=0):
    """Calculates Accuracy, MSE, MAE, RSS, AIC/BIC"""
    # Convert probabilities to binary class (Threshold = 0.5)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_prob)
    mae = mean_absolute_error(y_true, y_prob)
    rss = np.sum((y_true - y_prob) ** 2)

    print(f"\n[{model_name} Performance Metrics]")
    print(f"  Accuracy: {acc:.2%}")
    print(f"  MSE:      {mse:.4f}")
    print(f"  MAE:      {mae:.4f}")
    print(f"  RSS:      {rss:.4f}")

    # Calculate AIC/BIC for Logistic Regression only (Probabilistic)
    if model_name == "Logistic Regression":
        ll = -log_loss(y_true, y_prob, normalize=False)
        k = n_features + 1  # Features + Intercept
        aic = 2 * k - 2 * ll
        bic = k * np.log(n_samples) - 2 * ll
        print(f"  AIC:      {aic:.4f}")
        print(f"  BIC:      {bic:.4f}")

    return acc


def perform_prediction(df):
    print("\n--- [Phase 4] Predictive Modeling & Advanced Validation ---")

    if "is_canceled" not in df.columns:
        print("  >> Target 'is_canceled' not found. Skipping Phase 4.")
        return

    # 1. Preprocessing
    df_model = df.copy()

    # Remove leakage or non-predictive columns
    drop_cols = ['reservation_status', 'reservation_status_date', 'arrival_date', 'departure_date', 'name', 'email',
                 'phone-number', 'credit_card']
    for c in drop_cols:
        if c in df_model.columns:
            df_model.drop(c, axis=1, inplace=True)

    # Label Encode Categoricals
    cat_cols = df_model.select_dtypes(include=['object', 'category']).columns
    print(f"  >> Encoding {len(cat_cols)} categorical features...")
    for col in cat_cols:
        df_model[col] = LabelEncoder().fit_transform(df_model[col].astype(str))

    # Split Data
    X = df_model.drop("is_canceled", axis=1).fillna(0)
    y = df_model["is_canceled"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ----------------------------------------
    # Model 1: Logistic Regression (Baseline)
    # ----------------------------------------
    print("  >> Training Baseline (Logistic Regression)...")
    lr = LogisticRegression(max_iter=5000)  # High max_iter to fix convergence
    lr.fit(X_train, y_train)
    y_pred_lr, y_prob_lr = predict_with_progress(lr, X_test, task_name="LR Inference")

    calculate_metrics(y_test, y_prob_lr, "Logistic Regression", n_features=X_train.shape[1], n_samples=len(y_test))

    # ----------------------------------------
    # Model 2: Random Forest (Proposed)
    # ----------------------------------------
    print("  >> Training Proposed Model (Random Forest)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf, y_prob_rf = predict_with_progress(rf, X_test, task_name="RF Inference")

    calculate_metrics(y_test, y_prob_rf, "Random Forest")

    # ----------------------------------------
    # Visualizations
    # ----------------------------------------

    # 1. ROC Curve Comparison
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_lr, tpr_lr, label=f"Baseline (AUC={auc(fpr_lr, tpr_lr):.2f})", linestyle='--')
    plt.plot(fpr_rf, tpr_rf, label=f"Proposed Model (AUC={auc(fpr_rf, tpr_rf):.2f})", linewidth=2, color='darkgreen')
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "7_roc_comparison.png"))
    plt.close()

    # 2. Precision-Recall Curve (Good for Imbalanced Data)
    precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_prob_lr)
    precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_prob_rf)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_lr, precision_lr, linestyle='--',
             label=f'Baseline (AP={average_precision_score(y_test, y_prob_lr):.2f})')
    plt.plot(recall_rf, precision_rf, color='darkgreen', linewidth=2,
             label=f'Proposed Model (AP={average_precision_score(y_test, y_prob_rf):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Robust Validation)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "8_precision_recall_curve.png"))
    plt.close()

    # 3. Feature Importance
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feat_imp[:10].plot(kind="barh", color='teal')
    plt.title("Top 10 Drivers of Cancellation (RF)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "9_feature_importance.png"))
    plt.close()

    # 4. Confusion Matrix
    cm = confusion_matrix(y_test, (y_prob_rf >= 0.5).astype(int))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix (Random Forest)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_confusion_matrix.png"))
    plt.close()

    # 5. Learning Curve (Overfitting Check)
    # Note: Takes a bit longer to run
    plot_learning_curve(rf, X_train, y_train, "Random Forest", "11_learning_curve.png")


# ------------------------------
# 7. MAIN EXECUTION
# ------------------------------
if __name__ == "__main__":
    DATA_FILE = "hotel_bookings_new.csv"

    df_clean = load_and_clean_data(DATA_FILE)

    if df_clean is not None:
        df_eda = perform_eda(df_clean)
        df_clustered = perform_clustering(df_eda)
        perform_prediction(df_clustered)

    print(f"\nSUCCESS: Pipeline Complete. All 14 Plots saved to '{OUTPUT_DIR}'")