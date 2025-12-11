# Socio-Economic Profiling of Tourists: Hybrid ML Approach 🏨

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Completed](https://img.shields.io/badge/Status-Distinction%20Grade-success)](https://github.com/erwin-shrestha)

**Author:** Erwin Shrestha  
**Module:** Machine Learning (STW7072CEM)  
**Institution:** Softwarica College of IT & E-Commerce (Coventry University)

---

## 📖 Overview

This project tackles the **37% cancellation rate** in the Nepalese hospitality sector using a **Hybrid Machine Learning Architecture**.

Instead of treating all bookings the same, this system first **segments tourists into four “Socio-Economic Personas”** using **K-Means**, and then feeds these personas into a **Random Forest model** to predict cancellation risk.

### 🚀 Key Features
- **Hybrid Pipeline:** K-Means (Profiling) + Random Forest (Prediction)
- **Automated EDA:** 14+ visualizations (correlation heatmaps, histograms, distributions)
- **Advanced Metrics:** AIC, BIC, Silhouette Score, ROC-AUC
- **Business Intelligence:** Feature importance → actionable managerial insights

---

## 🛠️ Tech Stack & Methodology

The workflow follows a 4-stage ML pipeline:

### **1. Data Preprocessing**
- Handling missing values  
- Duplicate removal  
- Feature engineering (`daily_spending_usd`, `total_nights`)  
- Label encoding (categoricals)

### **2. Exploratory Data Analysis (EDA)**
- Target balance  
- Histogram distributions  
- Multicollinearity analysis (correlation matrix)  
- Missing value heatmap  

### **3. Unsupervised Learning (Profiling)**
- Algorithm: **K-Means (k = 4)**
- Validation: **Elbow Method**, **Silhouette Score**
- Output: Four distinct socio-economic personas

### **4. Supervised Learning (Prediction)**
- Baseline: **Logistic Regression**
- Proposed: **Random Forest Classifier**
- Validation: ROC-AUC, Confusion Matrix, Learning Curve, Precision–Recall Curve  

---

## 📊 Results Summary

| Metric | Logistic Regression | Random Forest | Improvement |
|-------|----------------------|---------------|-------------|
| **Accuracy** | 79.39% | **82.70%** | **+3.31%** |
| **MSE** | 0.1441 | **0.1221** | **-15.2%** |
| **AUC** | 0.86 | **0.90** | **+0.04** |

> **Key Finding:** *Lead Time* and *Deposit Type* were the two strongest predictors of cancellation risk.

---

## 📂 Repository Structure

```
├── app.py                        # Main ML pipeline
├── MachineLearningAssignment.pdf # Research paper
├── hotel_bookings_new.csv        # Dataset (user-provided)
├── requirements.txt              # Dependencies
├── output_plots/                 # Auto-generated visualizations
└── README.md
```

---

## 🔧 Installation

### **1. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate         # Windows
```

### **2. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Pipeline

Ensure your dataset is named:

```
hotel_bookings_new.csv
```

Run the script:

```bash
python app.py
```

The pipeline will automatically:

1. Clean & preprocess data  
2. Perform EDA  
3. Cluster tourists with K-Means  
4. Train Logistic Regression baseline  
5. Train Random Forest (proposed model)  
6. Save **all plots** to `output_plots/`

---

## 🧠 Tourist Persona Clustering

The **4 socio-economic personas** identified by K-Means:

- **Cluster 0 – Budget Planners**
- **Cluster 1 – Standard Travelers**
- **Cluster 2 – Mid-range Guests**
- **Cluster 3 – High-Value Luxury Travelers**

These personas significantly boost prediction performance.

---

## 🗂 Output Files (Auto-generated)

All visualizations are stored in:

```
output_plots/
```

Includes:
- Target distribution  
- Feature histograms  
- Correlation matrix  
- Missing value heatmap  
- Elbow method  
- Silhouette validation  
- Cluster scatter plot  
- Cluster spending profiles  
- ROC curve  
- Precision–Recall curve  
- Feature importance  
- Confusion Matrix  
- Learning Curve  

---

## 🛠 Technologies Used

- Python  
- Pandas / NumPy  
- Scikit-Learn  
- Matplotlib  
- Seaborn  
- Statsmodels  

---

## 📜 Research Paper

For full methodology and results:

```
MachineLearningAssignment.pdf
```

---

## 📬 Contact

**Author:** Erwin Shrestha  
📧 Email: erwin.shrestha17@gmail.com  
🐙 GitHub: https://github.com/erwinshrestha17

---

⭐ *If you found this project helpful, feel free to star the repository!*
