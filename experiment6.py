import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X6 = cancer.data
y6 = cancer.target

params = {'C':[0.1,1,10], 'gamma':[0.01,0.1,1]}

grid = GridSearchCV(SVC(kernel='rbf'), params)
grid.fit(X6, y6)

print("Best Params:", grid.best_params_)