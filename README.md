# Predicting Student Dropout and Academic Success

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-0f766e?style=for-the-badge)](https://akram23679.github.io/Student-Dropout-Prediction/)

**Machine Learning · Multi-class Classification · Akram Ali**

A machine learning project that predicts whether a student will **drop out**, stay **enrolled** or **graduate**, using academic, demographic and socio-economic data for **4,424 students**. Six classification models were trained, tuned and compared to find the strongest early-warning signals for at-risk students.

This repository contains the Jupyter notebook, the dataset and a responsive portfolio webpage that presents the analysis with interactive charts.

---

## Live Demo

**[View the live project →](https://akram23679.github.io/Student-Dropout-Prediction/)**

---

## Project Overview

| Item | Detail |
|---|---|
| Students | 4,424 |
| Input features | 34 |
| Target classes | Dropout (32.1%), Enrolled (17.9%), Graduate (49.9%) |
| Train / test split | 80 / 20, stratified (3,539 / 885) |
| Models compared | 6 |
| Best test accuracy | 77.7% (tuned SVM) |
| Best macro F1 | 0.712 (tuned SVM) |

**Dataset:** *Predict Students' Dropout and Academic Success*, Realinho et al., UCI Machine Learning Repository.

---

## Tools & Technologies

| Tool | Purpose |
|---|---|
| Python, Pandas, NumPy | Data loading, inspection and preparation |
| Matplotlib, Seaborn | Exploratory data analysis and charts |
| Scikit-learn | Pipelines, models, cross-validation and metrics |
| Jupyter Notebook | Analysis workflow |
| HTML / CSS / JavaScript, Chart.js | Portfolio webpage |

---

## Method

1. Loaded the CSV (with BOM handling) and checked types, missing values and duplicates. There were none.
2. Explored the target distribution, feature histograms, the correlation matrix, macroeconomic indicators and academic performance by outcome.
3. Split the data 80/20 with stratification to keep class proportions.
4. Wrapped scale-sensitive models in a `StandardScaler` pipeline so no information leaks from the test set.
5. Tuned KNN (`k`) and SVM (`C`, `gamma`) with 5-fold stratified cross-validation on training data only.
6. Evaluated every model on the held-out test set using accuracy and **macro F1**, because the classes are imbalanced.

---

## Model Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| **SVM (tuned, C = 100, gamma = 0.001)** | **0.777** | **0.712** |
| Random Forest (200 trees) | 0.774 | 0.706 |
| Logistic Regression | 0.773 | 0.696 |
| SVM (default) | 0.755 | 0.670 |
| KNN (k = 14) | 0.716 | 0.612 |
| Decision Tree | 0.705 | 0.649 |

---

## Key Findings

1. **Three models effectively tie** – tuned SVM, Random Forest and Logistic Regression are within 0.4 points (about 4 of 885 predictions).
2. **Enrolled is the hardest class** – the best model recalls only 43% of Enrolled students, most of whom are predicted as Graduate.
3. **Semester results lead everything** – dropouts pass 1.9 units in the 2nd semester on average versus 6.2 for graduates. Approved units and grades are the top four predictors.
4. **Overdue fees signal dropout** – 86.6% of students with tuition fees not up to date dropped out, versus 24.7% of those who were current.
5. **Scholarships align with graduation** – 76% of scholarship holders graduated, against 41% of other students.
6. **Older entrants are more at risk** – students enrolling at 25–29 dropped out at 57.6%, versus 21.2% for ages 17–20.

These are associations in the dataset, not proof of cause.

---

## Recommendations

1. Flag at-risk students after the 1st semester using approved units and grades
2. Treat overdue fees and debt as early alerts and offer financial support
3. Review scholarship reach and eligibility
4. Provide targeted support for mature students
5. Improve recall on the Enrolled class with class weighting, resampling or gradient boosting
6. Build an enrolment-time model for earlier warnings

---

## Repository Structure

```
├── index.html                                   # Portfolio page: content, charts and interactivity
├── styles.css                                   # Page styling, light/dark themes and responsive layout
├── Predict_Student_Dropout_And_Acad.ipynb       # Full analysis notebook
├── dataset.csv                                  # Student dataset (UCI)
└── README.md                                    # Project documentation
```

---

## Run Locally

**Webpage:** open `index.html` in any modern browser. An internet connection is needed to load Google Fonts and Chart.js.

**Notebook:**
```bash
git clone https://github.com/akram23679/Student-Dropout-Prediction.git
cd Student-Dropout-Prediction
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Predict_Student_Dropout_And_Acad.ipynb
```

## Deploy with GitHub Pages

1. Push the files to your repository.
2. Go to **Settings → Pages**.
3. Under **Source**, choose the `main` branch and the `/ (root)` folder.
4. Save. The site will be live at https://akram23679.github.io/Student-Dropout-Prediction/.

---

## Webpage Features

- Responsive layout for desktop, tablet and mobile
- Light / dark theme toggle (follows system setting by default)
- Tabbed summary card (Dataset, Models, Drivers)
- Interactive charts: outcome mix, academic performance, financial and demographic patterns
- Interactive correlation heatmap of all 34 features with hover details
- Model comparison, per-class F1, confusion matrix and feature importance

---

## Limitations

- Data comes from a single higher-education institution.
- The strongest features are semester results, available only after students start.
- Categorical codes were used as numbers rather than one-hot encoded.
- Demographic and financial patterns show associations, not causes.

---

## Author

**Akram Ali**  
Data Analyst  
GitHub: [@akram23679](https://github.com/akram23679)
