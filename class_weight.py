from matplotlib import pyplot as plt
from sklearn.base import ClassifierMixin, MultiOutputMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

iteration = 100
#load_data
X, y = load_digits(return_X_y=True)

#train_test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train 1 without class weights
model = LogisticRegression()
model.fit(x_train, y_train)

#prediction
y_pred = model.predict(x_train)

wrong = y_train[y_train != y_pred]

sample_weight = np.where(y_train != y_pred, 2, 1)

class_no = list(range(10))

# model = LogisticRegression(class_weight=class_weight)
model.fit(x_train, y_train, sample_weight=sample_weight)


