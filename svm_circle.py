import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles

X, y = make_circles(
    n_samples=100,
    factor=0.3,
    noise=0.1,
    random_state=42)

y = 2 * y - 1

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
titles = ['Linear Kernel', 'Poly Kernel', 'RBF Kernel', 'Sigmoid Kernel']

fix, axes = plt.subplots(2, 2, figsize=(12, 10))

axes = axes.ravel()

for i, kernel in enumerate(kernels):
    clf = SVC(kernel=kernel, C=1, degree=3, gamma='auto', random_state=42)
    clf.fit(X, y)

    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
    )

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axes[i].contourf(xx,yy,Z, level=np.linspace(Z.min(), Z.max(), 50), cmap='coolwarm', alpha=0.8)
    axes[i].contour(xx,yy,Z, level=np.linspace(Z.min(), Z.max(), 50))
    axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k')
    axes[i].set_xlim(X[:, 0].min() -1 , X[:, 1].max()+1)
    axes[i].set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

    axes[i].set_title(titles[i])

plt.tight_layout()
plt.show()