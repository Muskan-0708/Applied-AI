from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC   # <-- add this
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# Dataset
X5, y5 = make_classification(n_samples=500, n_features=10)

# Models
models = {
    'LR': LogisticRegression(),
    'SVM': SVC(probability=True),
    'RF': RandomForestClassifier()
}

# Training & Evaluation
for name, model in models.items():
    model.fit(X5, y5)
    preds = model.predict(X5)
    print(name, "Accuracy:", accuracy_score(y5, preds))