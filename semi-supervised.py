import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import classification_report

X, y = make_blobs(
    n_samples=60,
    n_features=2,
    centers=3,
    cluster_std=3,
    random_state=42
)

LabelSpreading()


unlabeled_rnd_idx = np.random.permutation(np.arange(60))[:40]
labeled_rnd_idx = np.random.permutation(np.arange(60))[40:]

X_labeled, y_labeled = X[labeled_rnd_idx], y[labeled_rnd_idx]
X_unlabeled, y_unlabeled = X[unlabeled_rnd_idx], y[unlabeled_rnd_idx]

model = LabelPropagation(kernel="rbf", gamma=0.01,max_iter=10000)
model.fit(X_labeled, y_labeled)
y_pred = model.predict(X_unlabeled)

print(classification_report(y_unlabeled, y_pred))

print(np.sum(y_unlabeled != y_pred))
#
# plt.subplot(121)
# plt.scatter(X[:, 0], X[:, 1], c=y)
#
# plt.subplot(122)
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()