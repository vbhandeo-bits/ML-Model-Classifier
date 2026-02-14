# model/training_notebook.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

# 1. Load Dataset
df = pd.read_csv('../WA_Fn-UseC_-HR-Employee-Attrition.csv')

# 2. Preprocessing
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode categorical features
for column in X.columns:
    if X[column].dtype == 'object':
        X[column] = LabelEncoder().fit_transform(X[column].astype(str))

if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y.astype(str))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Training & Evaluation of 6 Models (Mandatory)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "kNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier()
}

print(f"{'Model':<25} | {'Accuracy':<10} | {'MCC':<10}")
print("-" * 50)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    print(f"{name:<25} | {acc:.4f}     | {mcc:.4f}")