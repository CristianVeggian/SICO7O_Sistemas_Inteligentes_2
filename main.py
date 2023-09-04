from Adaline import Adaline
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

newY = list()

for classe in y:
    if classe == 0:
        newY.append(0)
    else:
        newY.append(1)

ada = Adaline(MAX_ITER=1000, n=0.001)
ada.treina(X[:120],newY[:120])
plt.plot(ada.erro[1:])
plt.show()
print(ada.predicao(X[120:]))