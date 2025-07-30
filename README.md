# 🧠 Stroke Prediction using Machine Learning

This project uses a health dataset to predict whether a person is likely to suffer from a **stroke** using machine learning classification models.

---

## 📂 Dataset

- **Source**: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Features**:
  - `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`
  - `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`
  - `smoking_status`
- **Target**:
  - `stroke`: 1 = Yes, 0 = No

---

## ⚙️ Tools & Libraries

- **Python Core**
- `pandas`, `numpy` for data manipulation
- `matplotlib`, `seaborn` for visualization
- `scikit-learn` for model building and evaluation

---

## 🧪 Models Compared

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Naive Bayes (GaussianNB)

---

## 🛠 Workflow

1. Loaded the dataset and handled missing values
2. Encoded categorical variables using one-hot encoding
3. Normalized numerical features (`age`, `glucose`, `bmi`)
4. Split data into train and test sets
5. Trained three classifiers
6. Evaluated performance using accuracy, confusion matrix, and classification report
7. Visualized confusion matrices and model accuracy

---

### 📌 Confusion Matrices

Each model’s confusion matrix was visualized to better understand prediction performance.

---

## ✍️ Author Note

This project was first built by me manually. Later, I used **AI support to improve parts of the logic and visualizations** for a cleaner and more professional structure.  
The goal was both learning and sharing something valuable.

---

## 🚀 Run the Project

1. Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn

