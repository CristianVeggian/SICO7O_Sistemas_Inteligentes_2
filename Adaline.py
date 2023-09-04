import numpy as np
import numpy.typing as npt
import random

import Extras as ex

class Adaline:
    
    def __init__(self, MAX_ITER = 100, bias = 1, n = 0.1, tol = 0.05):
        self.MAX_ITER = MAX_ITER
        self.bias = bias
        self.n = n
        self.tol = tol

    def degrau(self, v):
        if v >= 0:
            return 1
        return 0

        #ativação linear
    def phi(self, v):
        return v

    def treina(self, X: list | npt.ArrayLike, d: list | npt.ArrayLike):

        self.W = np.array([random.random() for _ in range(len(X[0])+1)])

        #normalizando os atributos de X
        X = ex.normalize(X)
        
        #adiciona coluna do bias
        X = np.insert(X, 0, [self.bias for _ in range(len(X))], axis=1)
        
        if type(d) is list:
            d = np.array(d)
                    
        epocas = 0
        self.erro = [1]
        
        while self.erro[-1] > self.tol and epocas < self.MAX_ITER: 
            counterD = -1
            #muda a ordem das amostras
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
                #atualiza TODOS os pesos sinapticos, para TODAS as amostras
                self.W = self.W + self.n*(d[counterD]-v)*line
            epocas += 1
            #erro quadrático: (d-X*W)^2/N
            self.erro.append((np.square(d - X*self.W)).mean())

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
            saida.append(self.degrau(v))
        return saida