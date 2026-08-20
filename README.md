# 🎓 Predict Student Dropout and Academic Success

A machine learning project that predicts whether a student will **Dropout**, remain **Enrolled**, or **Graduate**, based on academic, demographic, and socio-economic data collected at the time of enrollment and after the first two semesters.

## 📌 Overview

Student dropout is a major concern for higher education institutions. This project builds and compares several classification models to predict student outcomes early, so institutions can identify at-risk students and intervene before it's too late.

The target variable has three classes:
- **Dropout**
- **Enrolled**
- **Graduate**

## 📂 Dataset

The dataset (`dataset.csv`) contains **4,424 student records** with **35 features**, including:

- **Demographics** — Marital status, Nationality, Gender, Age at enrollment, Displaced, International
- **Academic background** — Application mode/order, Course, Daytime/evening attendance, Previous qualification, Parents' qualification & occupation
- **Financial & social factors** — Debtor, Tuition fees up to date, Scholarship holder, Educational special needs
- **Academic performance** — Curricular units enrolled/approved/evaluated/graded for the 1st and 2nd semesters
- **Macroeconomic indicators** — Unemployment rate, Inflation rate, GDP
- **Target** — Dropout / Enrolled / Graduate

## 🔍 Exploratory Data Analysis (EDA)

Before modeling, the notebook performs EDA using **Pandas**, **NumPy**, **Matplotlib**, and **Seaborn** to understand feature distributions and their relationship with the target:

- `df.head()`, `df.info()`, `df.describe()` for a first look at the data
- Histograms of all numerical features
- Distribution plots such as `Unemployment rate` and `Inflation rate` segmented by `Target`, to see how macroeconomic conditions relate to student outcomes

## 🤖 Models & Results

The data was split 80/20 into training and test sets (`train_test_split`, `random_state=101`). Several classifiers from **scikit-learn** were trained and evaluated on accuracy, precision, recall, and F1-score:

| Model | Accuracy |
|---|---|
| Logistic Regression | **77.4%** |
| K-Nearest Neighbors (k=1) | 66.0% |
| K-Nearest Neighbors (tuned, k=11) | 69.9% |
| Decision Tree | 71.1% |
| **Random Forest (n_estimators=200)** | **78.5%** ✅ Best performer |
| Support Vector Machine (SVC) | 75.0% |
| SVM (tuned via GridSearchCV) | 76.6% |

**Random Forest achieved the highest accuracy (78.5%)**, closely followed by Logistic Regression (77.4%). Across all models, the **Graduate** class was predicted most reliably, while the **Enrolled** class was consistently the hardest to classify due to class overlap with Dropout/Graduate.

## 🛠️ Tech Stack

- **Python 3**
- **NumPy** & **Pandas** — data manipulation
- **Matplotlib** & **Seaborn** — visualization
- **scikit-learn** — modeling & evaluation (`LogisticRegression`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`, `SVC`, `GridSearchCV`)
- **Google Colab** — development environment

## 📁 Repository Structure

```
.
├── Predict_Student_Dropout_And_Acad.ipynb   # Main notebook: EDA, modeling, evaluation
├── dataset.csv                              # Student records dataset
└── README.md                                # Project documentation
```

## 🚀 Getting Started

1. Clone the repository
   ```bash
   git clone https://github.com/Akram23679/Predict_Student_Dropout_And_Acad.git
   cd Predict_Student_Dropout_And_Acad
   ```
2. Install dependencies
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Open and run the notebook
   ```bash
   jupyter notebook Predict_Student_Dropout_And_Acad.ipynb
   ```
   *(Originally developed in Google Colab — `google.colab.files` upload calls can be swapped for a direct `pd.read_csv("dataset.csv")` when running locally.)*

## 📈 Key Takeaways

- Ensemble methods (Random Forest) and simple linear models (Logistic Regression) outperformed distance-based methods (KNN) on this dataset.
- First and second semester academic performance (units approved, grades) were strong predictors of final outcome.
- Macroeconomic indicators (unemployment, inflation, GDP) show visible but secondary influence on dropout risk.

## 📄 License

This project is open source and available for educational and research purposes.
