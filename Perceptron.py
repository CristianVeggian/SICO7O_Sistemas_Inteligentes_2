import numpy as np
import numpy.typing as npt
import random

class Perceptron:
    
    def __init__(self, MAX_ITER = 100, bias = 1, n = 0.1):
        self.MAX_ITER = MAX_ITER
        self.bias = bias
        self.n = n

    def phi(self, v):
        if v >= 0:
            return 1
        return 0

    def treina(self, X: list | npt.ArrayLike, d: list | npt.ArrayLike):

        self.W = np.array([random.random() for _ in range(len(X)-1)])

        if type(X) is list:
            X = np.array(X) 
        
        if type(d) is list:
            d = np.array(d)
                    
        epocas = 0
        erro = True
        
        while erro and epocas < self.MAX_ITER:
            epocas += 1
            erro = False
            counterD = -1
            for line in X:
                counterD += 1
                counterX = 0
                v = 0
                while counterX < len(line):
                    v += line[counterX]*self.W[counterX]
                    counterX += 1
                v += self.bias * self.W[-1] 
                v = self.phi(v)
                if v != d[counterD]:
                    self.W = self.W + self.n*(d[counterD]-v)*np.append(line, self.bias)
                    erro = True

    def predicao(self, X: list | npt.ArrayLike):
        if type(X) is list:
            X = np.array(X) 
            
        saida = list()
            
        for line in X:
            counterX = 0
            v = 0
            while counterX < len(line):
                v += line[counterX]*self.W[counterX]
                counterX += 1
            v += self.bias * self.W[-1] 
            saida.append(self.phi(v))
            
        return saida

pct = Perceptron()
pct.treina([[0,0],[0,1],[1,0],[1,1]], [[0],[1],[1],[1]])
print(pct.predicao([[-1,1],[1,-1],[0,-1],[3,2]]))
