## Titanic Survival Prediction: Machine Learning Analysis 
## Author: Adeyeye Blessing Temidayo
## CIN: PIT/DEC25/10676 
Repository: 
PRODIGY_DS_02

## 📌 Project Overview
This project performs a comprehensive analysis of the Titanic dataset to predict passenger survival using machine learning.
The project demonstrates the full data science lifecycle, from initial data cleaning and exploratory analysis to advanced
modeling and performance evaluation. By comparing Logistic Regression and XGBoost, this study highlights the trade-offs
between model interpretability and predictive power.

## 📊 Dataset Information
Source: Prodigy InfoTech / Kaggle Titanic Dataset
Records: 891 passengers
Original Features: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
Engineered Features: Has_Cabin, Title, Family_Size, Age_Group, Fare_Category, Is_Alone

## 🎯 Key Objectives
Perform thorough Data Cleaning and Feature Engineering.

Conduct Exploratory Data Analysis (EDA) to identify key predictors of survival.

Handle class imbalance using SMOTE.

Develop and compare predictive models (Logistic Regression vs. XGBoost).

Evaluate models using robust metrics like ROC-AUC, Precision, and Cross-Validation.

## 🛠️ Technical Implementation 
Data Pipeline Preprocessing: Handled missing values (Age, Embarked), encoded categorical variables, and applied feature scaling.

Imbalance Handling: Utilized SMOTE (Synthetic Minority Over-sampling Technique) to ensure the model wasn't biased toward the majority class.

Modeling: * Logistic Regression: Used as a high-interpretability baseline.XGBoost Classifier: Used to capture non-linear relationships and complex patterns.

Evaluation: Metrics include Accuracy, ROC-AUC (Primary), Average Precision, and Confusion Matrices.

## 📊 Data Dictionary
| Variable | Definition | Key/Notes |
| :--- | :--- | :--- |
| **Survived** | Survival Status | 0 = No, 1 = Yes |
| **Pclass** | Ticket Class | 1 = 1st, 2 = 2nd, 3 = 3rd |
| **Sex** | Gender | Male, Female |
| **Age** | Age in years | Fractional if less than 1 |
| **SibSp** | # of siblings / spouses aboard | - |
| **Parch** | # of parents / children aboard | - |
| **Fare** | Passenger fare | - |
| **Embarked** | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |



### 📊 Model Performance Summary
| Metric | Logistic Regression | XGBoost (Final Model) |
| :--- | :--- | :--- |
| **Mean CV ROC-AUC** | 0.8576 | 0.8473 |
| **Test Set ROC-AUC** | 0.8621 | **0.8747** |
| **Strengths** | Interpretability & Stability | High Discriminative Power 

## Survival Distribution
![Correlation Matrix](figures/Features Correlation Matrix.png)

Key Findings

XGBoost emerged as the final model due to its superior performance on unseen data.

SMOTE significantly improved the model's ability to identify survivors.

Feature interactions (like Class vs. Age) play a critical role in predicting outcomes.

## ⚠️ Limitations 
Feature Constraints: Historical data lacks specific details like exact cabin proximity to lifeboats.

SMOTE Bias: Synthetic data may not perfectly represent real-world nuances.

Interpretability: XGBoost is a "black box" compared to Logistic Regression, requiring tools like SHAP for full transparency.

## 🚀 Future Work 

Model Explainability: Implement SHAP or LIME to explain individual predictions.

Hyperparameter Tuning: Use Bayesian Optimization for further XGBoost refinement.

Deployment: Wrap the model in a Flask or FastAPI wrapper for real-time predictions.

Ensemble Learning: Experiment with Stacking models to combine the strengths of linear and tree-based methods.

📂 Project Structure 

Plaintext├── Titanic_Analysed.ipynb      # Main analysis & modeling notebook
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── data/                       # Dataset folder (if applicable)


## 💻 Installation & Usage
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Temidayo23/PRODIGY_DS_02](https://github.com/Temidayo23/PRODIGY_DS_02)
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Analysis:**
    Open `Titanic_Analysed.ipynb` in Jupyter Notebook and run all cells.

🧰 Technologies UsedLanguage: Python 3.8+Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Imbalanced-learn (SMOTE)

## 📧 Contact
Adeyeye Blessing Temidayo (adeyeyeblessing2017@gmail.com) Feel free to reach out for collaborations or questions regarding this analysis!

## 📄 License
MIT License Copyright (c) 2026 Adeyeye Blessing Temidayo

Permission is hereby granted, free of charge... (See full text in repository)

Disclaimer: This project was developed as part of a Data Science project with Prodigy InfoTech and uses the Kaggle Twitter Entity Sentiment Analysis Dataset.






