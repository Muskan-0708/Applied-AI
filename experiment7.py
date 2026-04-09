import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

X7, y7 = make_classification(n_samples=500, n_features=10)

rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X7, y7)

print("OOB Score:", rf.oob_score_)

# Feature Importance
plt.bar(range(len(rf.feature_importances_)), rf.feature_importances_)
plt.title('Feature Importance')
plt.show()