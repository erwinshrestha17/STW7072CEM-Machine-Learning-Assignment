# Socio-Economic Profiling of Tourists: Hybrid ML Approach 🏨

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Completed](https://img.shields.io/badge/Status-Distinction%20Grade-success)](https://github.com/erwinshrestha17)

**Author:** Erwin Shrestha  
**Module:** Machine Learning (STW7072CEM)  
**Institution:** Softwarica College of IT & E-Commerce, Coventry University  
**Project Title:** Socio-Economic Profiling of Tourists: A Hybrid K-Means and Random Forest Approach to Cancellation Prediction in the Nepalese Hospitality Sector

---

## 📖 Overview

This project addresses the hotel booking cancellation problem in the Nepalese hospitality sector using a **hybrid machine learning architecture**.

Nepal’s tourism market is highly seasonal, so cancelled bookings can directly affect hotel revenue, staffing, inventory planning, and room availability. Instead of treating all bookings the same, this system first segments tourists into socio-economic personas using **K-Means Clustering**, then uses those personas as an additional feature in a **Random Forest Classifier** to predict cancellation risk.

The final hybrid model improves performance compared to a standard Logistic Regression baseline.

---

## 🎯 Research Aim

The main aim of this assignment is to move beyond manual guessing and build a data-driven cancellation prediction system that can help hotels identify high-risk bookings and make better revenue management decisions.

---

## ✅ Objectives

- Explore hotel booking data and identify key cancellation patterns.
- Use clustering to create tourist personas based on spending and booking behavior.
- Build a supervised machine learning model to predict whether a booking will be cancelled.
- Compare the hybrid Random Forest model against a Logistic Regression baseline.
- Interpret the results for practical hotel revenue management decisions in Nepal.

---

## 🚀 Key Features

- **Hybrid ML Pipeline:** K-Means clustering for profiling + Random Forest for prediction.
- **Tourist Persona Segmentation:** Groups tourists into four socio-economic behavior clusters.
- **Exploratory Data Analysis:** Includes target distribution, lead time analysis, correlation heatmap, and spending patterns.
- **Model Comparison:** Logistic Regression baseline compared with Random Forest.
- **Evaluation Metrics:** Accuracy, MSE, AUC, ROC Curve, Confusion Matrix, Learning Curve, and Feature Importance.
- **Business Insights:** Helps hotels design smarter overbooking, deposit, and cancellation policies.

---

## 📊 Dataset

The project uses a hotel booking dataset containing approximately **120,000 records**.

Key variables include:

| Feature | Description |
|---|---|
| `lead_time` | Number of days between booking and arrival |
| `daily_spending_usd` | Estimated daily spending by the guest |
| `total_spending_usd` | Total estimated spending value |
| `adults` | Number of adults in the booking |
| `deposit_type` | Deposit condition for the booking |
| `is_canceled` | Target variable: `0` = not cancelled, `1` = cancelled |

The dataset shows an approximate **37% cancellation rate**, making cancellation prediction important for revenue protection.

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels

---

## 🧪 Methodology

The workflow follows a four-stage machine learning pipeline.

### 1. Data Cleaning and Preprocessing

The data preparation process includes:

- Handling missing values
- Removing duplicate records
- Encoding categorical variables
- Creating engineered features such as spending metrics
- Scaling numerical features for clustering
- Splitting the dataset into training and testing sets

Missing numerical values are handled using median imputation, while missing categorical values are marked as `Unknown`.

### 2. Exploratory Data Analysis

The EDA stage investigates:

- Distribution of cancellations
- Distribution of lead time
- Correlation between booking features
- Relationship between spending and cancellation behavior
- Missing value patterns
- Feature-level trends affecting cancellation risk

Important findings:

- Higher lead time is associated with higher cancellation risk.
- Higher spending is generally associated with lower cancellation risk.
- Deposit type and lead time are among the strongest predictors of cancellation.

### 3. K-Means Clustering

K-Means clustering is used to create tourist personas.

Features used for clustering:

```python
features = [
    "daily_spending_usd",
    "total_spending_usd",
    "lead_time",
    "adults"
]
```

The **Elbow Method** suggests that **k = 4** is the optimal number of clusters.

The clustering stage produces a new feature:

```python
Cluster_ID
```

This feature represents tourist behavior groups such as budget planners and high-value/luxury guests.

### 4. Random Forest Classification

After clustering, the generated `Cluster_ID` feature is added to the supervised learning dataset.

The main model configuration is:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

Random Forest was selected because it:

- Handles non-linear relationships well
- Works effectively with mixed feature types
- Reduces overfitting through ensemble learning
- Provides feature importance for model interpretation

---

## 📈 Model Performance

| Metric | Logistic Regression | Random Forest | Improvement |
|---|---:|---:|---:|
| **Accuracy** | 79.39% | **82.70%** | **+3.31%** |
| **MSE** | 0.1441 | **0.1221** | **-15.2%** |
| **AUC Score** | 0.86 | **0.90** | **+0.04** |
| **AIC** | 21434.7 | N/A | N/A |

The Random Forest model achieved the best performance with:

- **82.70% accuracy**
- **0.1221 Mean Squared Error**
- **0.90 AUC score**

This shows that adding tourist persona information improves cancellation prediction.

---

## 🧠 Tourist Persona Clustering

The K-Means model identifies four socio-economic tourist personas:

- **Cluster 0 – Budget Planners**
- **Cluster 1 – Standard Travelers**
- **Cluster 2 – Mid-range Guests**
- **Cluster 3 – High-Value Luxury Travelers**

These personas help the predictive model understand booking behavior more clearly than raw features alone.

---

## 🔍 Key Findings

- Lead time is one of the strongest indicators of cancellation risk.
- Deposit type strongly influences whether guests cancel.
- Spending behavior is useful for identifying different booking risk groups.
- Tourist personas improve model performance compared to using raw booking features only.
- The hybrid K-Means + Random Forest approach performs better than Logistic Regression.

---

## 🏨 Managerial Implications

This model can help hotel managers in Nepal make better decisions such as:

- Applying smarter overbooking strategies
- Identifying bookings with high cancellation risk
- Offering flexible cancellation policies to low-risk high-value guests
- Applying stricter deposit rules for high-risk booking groups
- Improving revenue forecasting during peak tourism seasons

Example cluster-based policy suggestions:

- **Budget Planner Segment:** Higher cancellation risk, suitable for stricter non-refundable policies on long lead-time bookings.
- **Luxury / High-Value Segment:** Lower cancellation risk, suitable for flexible cancellation benefits to attract valuable guests.

---

## ⚖️ Ethical Considerations

The project uses behavioral and spending-based features rather than sensitive demographic attributes such as gender, race, or religion. However, automated customer profiling still requires careful use.

Important ethical concerns include:

- Avoiding unfair pricing discrimination
- Monitoring bias in prediction outcomes
- Retraining the model when tourism patterns change
- Using predictions as decision-support, not as the only decision-maker

---

## 📂 Repository Structure

```text
STW7072CEM-Machine-Learning-Assignment/
│
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── hotel_bookings_new.csv
├── outputs/
│   ├── cancellation_distribution.png
│   ├── lead_time_distribution.png
│   ├── correlation_heatmap.png
│   ├── elbow_method.png
│   ├── cluster_spending_profile.png
│   ├── roc_curve.png
│   ├── learning_curve.png
│   ├── feature_importance.png
│   └── confusion_matrix.png
└── report/
    └── MachineLearningAssignmentReport.pdf
```

---

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/erwinshrestha17/STW7072CEM-Machine-Learning-Assignment.git
cd STW7072CEM-Machine-Learning-Assignment
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Make sure your dataset is available in the project directory or inside the `data/` folder.

Run the main analysis pipeline:

```bash
python app.py
```

The script should perform:

1. Data loading
2. Data cleaning
3. Exploratory data analysis
4. K-Means clustering
5. Logistic Regression baseline training
6. Random Forest model training
7. Model evaluation
8. Output generation for figures and metrics

---

## 🗂️ Output Files

Generated visualizations are saved in the `outputs/` folder.

Typical outputs include:

- Target distribution
- Lead time distribution
- Correlation heatmap
- Missing value heatmap
- Elbow method plot
- Silhouette validation plot
- Cluster spending profile
- ROC curve
- Precision–Recall curve
- Feature importance plot
- Confusion matrix
- Learning curve

---

## 📜 Research Paper

For the full methodology, figures, literature review, and discussion, refer to:

```text
report/MachineLearningAssignmentReport.pdf
```

---

## 🔮 Future Improvements

Possible future extensions include:

- Adding real-time weather data
- Including flight delay or travel disruption data
- Deploying the model as a web dashboard
- Adding SHAP explanations for model interpretability
- Testing XGBoost, LightGBM, or CatBoost
- Monitoring concept drift over time
- Building a hotel revenue management recommendation system

---

## 🔗 Source Code

The complete source code is available at:

```text
https://github.com/erwinshrestha17/STW7072CEM-Machine-Learning-Assignment
```

---

## 📬 Contact

**Author:** Erwin Shrestha  
**Email:** erwin.shrestha17@gmail.com  
**GitHub:** https://github.com/erwinshrestha17  
**Institution:** Softwarica College of IT & E-Commerce, Coventry University  
**Location:** Kathmandu, Nepal

---

## 📚 References

- Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel Booking Demand Datasets.
- Morales, L., & Wang, J. (2010). Forecasting cancellation rates for services booking revenue management.
- Breiman, L. (2001). Random Forests.
- Mitchell, T. M. (1997). Machine Learning.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.
- Scikit-learn Developers. User Guide: Clustering and Ensemble Methods.

---

⭐ If you found this project useful, feel free to star the repository.
