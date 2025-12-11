# 🏨 Tourism Analytics Project  
### Unsupervised Clustering • EDA • Feature Engineering • Predictive Modeling

This project analyzes **hotel booking data** using a full **machine learning pipeline** consisting of:

- Comprehensive Exploratory Data Analysis (EDA)
- Feature Engineering
- K-Means Clustering (Unsupervised Learning)
- Logistic Regression & Random Forest (Supervised Learning)
- Model comparison & performance evaluation
- Automated diagram generation

All generated plots are saved inside the `output_plots/` folder.

---

# 🚀 Project Workflow

## **1. Data Loading & Cleaning**
- Handles missing values  
- Encodes categorical features  
- Removes duplicates  
- Creates engineered features:
  - `daily_spending_usd`
  - `total_spending_usd`
  - `total_nights`

---

# 📊 **2. Exploratory Data Analysis (EDA)**  
The script automatically generates:

### 🔹 Missing Data Heatmap  
### 🔹 Numerical Distributions  
### 🔹 Outlier Boxplots  
### 🔹 Categorical Count Plots  
### 🔹 Full Correlation Heatmap  
### 🔹 VIF (Multicollinearity Check)  
### 🔹 Cancellation Rate by:
- Hotel  
- Market Segment  
- Country  

### 🔹 Lead-Time vs Cancellation Behavior  
### 🔹 Cramér’s V (Categorical → Target Correlation)  
### 🔹 Spending Tier Visualization  
### 🔹 Pairplot of Core Numerical Features  

All diagrams are exported to `output_plots/`.

---

# 🔍 **3. Clustering (Unsupervised Learning)**  
Performed using **K-Means** on socio-economic features:

- Daily spending  
- Total spending  
- Lead time  
- Week-night stays  
- Adults  

### Includes:
- Cluster visualization  
- Elbow method plot  
- Radar chart for profile comparison  

---

# 🤖 **4. Predictive Modeling**  
Models used:

- **Logistic Regression (Baseline)**
- **Random Forest (Proposed Model)**

### Evaluation Includes:
- Accuracy  
- Confusion Matrix  
- Classification Report  
- ROC Curve (AUC)  
- Feature Importance Ranking  

---

# 📂 Folder Structure

