# ⚖️ Class Imbalance Handling in Churn Prediction (Logistic Regression)
This project explores the impact of handling class imbalance using undersampling, SMOTE, and SMOTE-Tomek Links in a logistic
regression-based churn prediction task. The Iranian Churn Dataset is used to evaluate how different resampling techniques affect classification performance, especially on the minority class.

## 📊 Dataset
Source: UCI Machine Learning Repository – Iranian Churn Dataset

Target Variable: Churn

- 1 → Churned
- 0 → Did not churn

The dataset is imbalanced, with far fewer churned customers (minority class).


## 🚀 Project Workflow

1. Data Loading & Exploration
   
- Loaded the dataset using pandas
- Inspected class distribution using value_counts() and visualized with seaborn
- Checked for missing values

2. Train-Test Split
   
- Used train_test_split with stratify=y to preserve imbalance
- Test set: 20%

3. Baseline Model (No Resampling)
   
- Model: LogisticRegression(max_iter=2000)
- Evaluation: classification_report from sklearn.metrics

4. Class Imbalance Fixes
   
🔹 Random Undersampling (RandomUnderSampler)

- Reduces majority class to match minority
- May lose useful data

🔹 SMOTE (Synthetic Minority Oversampling Technique)

- Generates synthetic samples for the minority class
- Improves recall at cost of potential overfitting

🔹 SMOTE + Tomek Links (Hybrid Resampling)

- Combines oversampling (SMOTE) and data cleaning (Tomek)
- Balances the data and removes ambiguous samples


## 🧾 Results Summary

| Model                  | Accuracy | Precision (1) | Recall (1) | F1-Score (1) |
|------------------------|----------|---------------|------------|--------------|
| No Resampling          | 0.89     | 0.80          | 0.41       | 0.55         |
| Random Undersampling   | 0.81     | 0.45          | 0.84       | 0.58         |
| SMOTE                  | 0.81     | 0.44          | 0.85       | 0.58         |
| SMOTE + Tomek Links    | 0.81     | 0.45          | 0.86       | 0.59         |

📌 *Note: Precision, recall, and F1-score refer to the minority class (`Churn = 1`).*

## 🧪 Observations

- The baseline model has high precision but very low recall for churners.
- Undersampling and SMOTE significantly improved recall, allowing the model to catch more churners.
- SMOTE-Tomek slightly improved both precision and recall, offering a balanced trade-off.

## ⚙️ Installation

1. Clone the repo:

```bash
git clone https://github.com/yourusername/churn_class_imbalance_fix.git
cd churn_class_imbalance_fix
```
2. **Install dependencies:**:   
```commandline
pip install -r requirements.txt
```

📂 Files in This Repo

- churn_class_imbalance_fix.ipynb: Full notebook with all preprocessing, modeling, and evaluations
- requirements.txt: All required Python libraries (pandas, imblearn, scikit-learn, matplotlib, seaborn)

👨‍💻 Author

Yoseph Negash

📧 yosephn22@gmail.com

📅 2025
