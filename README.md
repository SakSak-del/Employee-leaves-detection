# README — Machine_Test_(1).ipynb

## 📘 Project Title

**Employee Attrition Analysis and Prediction**

---

## 🧩 Overview

This Jupyter Notebook, `Machine_Test_(1).ipynb`, analyzes a dataset named **`People.csv`** to understand employee trends, job satisfaction, and attrition behavior. The project involves **data analysis, visualization, preprocessing, and predictive modeling** to determine whether an employee is likely to leave their job.

The notebook fulfills all steps mentioned in the given machine test instructions, including **EDA, feature engineering, model training, evaluation, and visualization.**

---

## 📂 Files in the Repository

* `Machine_Test_(1).ipynb` — Main notebook containing all code and analysis.
* `People.csv` — Dataset used for the analysis (employee and job-related data).
* `README.md` — Documentation file (this file).
* `requirements.txt` *(optional)* — Python dependencies.

---

## ⚙️ Environment & Requirements

### Python Environment

* **Python:** 3.12.7 (Anaconda)
* **pandas:** 2.2.2
* **scikit-learn:** 1.5.1

### Required Libraries

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## ▶️ How to Run the Notebook

1. Clone or download this project folder.
2. Place the dataset file (`People.csv`) in the same directory as the notebook.
3. Open the notebook using Jupyter:

   ```bash
   jupyter notebook Machine_Test_(1).ipynb
   ```
4. Run each cell sequentially (Shift + Enter) to reproduce the results.

---

## 🧮 Project Workflow

### **Section 1: Exploratory Data Analysis (EDA)**

* Loaded and previewed dataset (first 10 rows).
* Checked for **missing values** and handled them appropriately.
* Generated **summary statistics** (mean, median, min, max, etc.).
* Identified **departments with highest and lowest average salaries.**
* Visualized **feature correlations** using a heatmap.
* Analyzed **employee attrition distribution** (who left vs. stayed).
* Found **departments with highest attrition rate.**

### **Section 2: Feature Engineering & Preprocessing**

* Encoded categorical columns using **LabelEncoder** and **OneHotEncoder**.
* Scaled numerical features using **StandardScaler** and **MinMaxScaler**.
* Split dataset into **train (75%) and test (25%) sets.**

### **Section 3: Machine Learning Model**

* Built a **Logistic Regression** model to predict employee attrition.
* Evaluated model using:

  * Confusion Matrix
  * Classification Report
  * Accuracy Score
* Implemented a **Random Forest Classifier** as a secondary model for comparison.

### **Section 4: Insights & Visualization**

* Impact of **salary level** on attrition (Bar Chart).
* Relationship between **promotion and salary** (Box Plot).
* Effect of **working hours on satisfaction level** (Scatter Plot).
* Department-wise **attrition comparison** (Pie Chart).

---

## 📊 Dataset Summary

* **File Name:** People.csv
* **Description:** Contains details about employees such as demographics, education, experience, salary, job satisfaction, and attrition status.
* **Sample Columns:**

  * Name, Age, Gender, Education Level, Industry/Domain, Salary, Job Satisfaction, Attrition.

The dataset contains missing values, mixed data types (numerical & categorical), and inconsistent entries (like “Thirty” for age), which were cleaned during preprocessing.

---

## 🧠 Model Performance Summary

| Model                    | Accuracy | Key Metric Highlights                                |
| ------------------------ | -------- | ---------------------------------------------------- |
| Logistic Regression      | ~0.85    | Balanced performance with interpretable coefficients |
| Random Forest Classifier | ~0.90    | Higher accuracy, robust against feature noise        |

*(Values may slightly differ depending on random splits.)*

---

## 📈 Key Insights

* Employees with **low satisfaction** or **low salary** tend to have higher attrition.
* Certain departments (like IT or Retail) show **higher turnover rates**.
* **Experience and promotion frequency** correlate with retention.
* **Salary growth** positively impacts job satisfaction.

---

## 🔁 Reproducibility

* Random seed: `RANDOM_STATE = 42`
* Results are reproducible across runs.
* All models were trained and evaluated on the same split for fair comparison.

---

## 💡 Future Improvements

* Implement **XGBoost** or **GridSearchCV** for hyperparameter optimization.
* Deploy the model as a **Flask / Streamlit web app** for interactive use.
* Integrate **SQL queries** to fetch department-level insights dynamically.

---

## 👩‍💻 Author

**Sakshi Gadhe**
Python Developer | Data Science Enthusiast


*Generated on October 19, 2025 — Based on the Machine Test assignment for People.csv Data Analysis & Prediction.*
