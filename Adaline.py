import numpy as np
import numpy.typing as npt
import random

import Extras as ex

class Adaline:
    
    def __init__(self, MAX_ITER = 100, bias = 1, n = 0.1, erro = 0.05):
        self.MAX_ITER = MAX_ITER
        self.bias = bias
        self.n = n
        self.erro = erro

    def phi(self, v):
        return v

    def treina(self, X: list | npt.ArrayLike, d: list | npt.ArrayLike):

        #self.W = np.array([random.random() for _ in range(len(X)-1)])
        self.W = [0.5,0.5,0.5]
        X = ex.normalize(X)
        
        if type(d) is list:
            d = np.array(d)
                    
        epocas = 0
        erro = 1
        
        while erro > self.erro and epocas < self.MAX_ITER: 
            counterD = -1
            np.random.shuffle(X)
            for line in X:
                counterD += 1
                counterX = 0
                v = 0
                while counterX < len(line):
                    v += line[counterX]*self.W[counterX]
                    counterX += 1
                v += self.bias * self.W[-1] 
                v = self.phi(v)
                self.W = self.W + self.n*(d[counterD]-v)*np.append(line, self.bias)
            epocas += 1
            XparaErro = list()
            for line in X:
                XparaErro.append(self.W*np.append(line, self.bias))
            XparaErro = np.array(XparaErro)
            erro = (np.square(d - self.W*XparaErro)).mean()

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

ada = Adaline()
ada.treina([[0,0],[0,1],[1,0],[1,1]],[[0],[0],[0],[1]])
print(ada.predicao([[1.1,1.1],[-0.1,-0.1]]))