import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load and prepare the dataset
df = pd.read_csv(r"C:\Users\saraye tel\OneDrive\Desktop\ARCH_Roadmap\Phase_1\healthcare-dataset-stroke-data.csv")

# Fill missing values
df = df.fillna(df.mean(numeric_only=True))

# Drop ID column if it's not informative
if "id" in df.columns:
    df = df.drop("id", axis=1)

# Encode categorical features
df = pd.get_dummies(df, columns=[
    'gender',
    'ever_married',
    'work_type',
    'Residence_type',
    'smoking_status'
], drop_first=True)

# Normalize numerical features
for col in ['age', 'avg_glucose_level', 'bmi']:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

# Prepare features and target variable
X = df.drop("stroke", axis=1).values
y = df["stroke"].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Initialize classifiers
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="linear"),
    "Naive Bayes": GaussianNB()
}

accuracy_scores = {}

# Training, prediction, evaluation
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    accuracy_scores[name] = acc

    print(f"\nModel: {name}")
    print(f"Accuracy: {acc:.2f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Stroke", "Stroke"],
                yticklabels=["No Stroke", "Stroke"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# Accuracy comparison chart
plt.figure(figsize=(8, 5))
sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()), palette="viridis")
plt.title("Accuracy Comparison of Models")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("Accuracy_Comparison.png", dpi=300)
plt.show()