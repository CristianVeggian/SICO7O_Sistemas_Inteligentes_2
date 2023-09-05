from Adaline import Adaline
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from Extras import confusionMatrix

iris = datasets.load_iris()
X = iris.data[:, [0,1]]
y = iris.target

newY = list()

for classe in y:
    if classe == 0:
        newY.append(-1)
    else:
        newY.append(1)

x_treino, x_teste, y_treino, y_teste = train_test_split(X,newY,test_size=0.2)

ada = Adaline(MAX_ITER=1000, n=0.001, tol=0.01)
ada.treina(x_treino, y_treino)
plt.plot(ada.erro)
plt.show()
y_predict = ada.predicao(x_teste)
print(confusionMatrix(y_teste, y_predict))