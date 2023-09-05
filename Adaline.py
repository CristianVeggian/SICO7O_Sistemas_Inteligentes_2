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
        return -1

        #ativação linear
    def phi(self, v):
        return v

    def treina(self, X: list | npt.ArrayLike, d: list | npt.ArrayLike):

        self.W = np.array([random.random() for _ in range(len(X[0])+1)])

        #adiciona coluna do bias
        X = np.insert(X, 0, [self.bias for _ in range(len(X))], axis=1)

        #normalizando os atributos de X
        X = ex.normalize(X)

        if type(d) is list:
            d = np.array(d)
                    
        epocas = 0
        self.erro = list()
        
        while epocas < self.MAX_ITER: 
            counter = 0
            #muda a ordem das amostras
            #np.random.shuffle(X)
            v = list()
            for line in X:
                v.append(np.sum(line*self.W))
                v = self.phi(v)
                #atualiza TODOS os pesos sinapticos, para TODAS as amostras
                self.W = self.W + self.n*(d[counter]-v[counter])*line
                counter += 1
            epocas += 1
            #erro quadrático: (d-X*W)^2/N
            self.erro.append((np.square(d - v)).mean())
            if self.erro[-1] < self.tol:
                break

    def predicao(self, X: list | npt.ArrayLike):

        #adiciona coluna do bias
        X = np.insert(X, 0, [self.bias for _ in range(len(X))], axis=1)
        #X = np.append(X, [[self.bias] for _ in range(len(X))], axis=1)

        #normalizando os atributos de X
        X = ex.normalize(X)
        saida = list()
            
        for line in X:
            v = line*self.W
            v = np.sum(v)
            saida.append(self.degrau(v))
        return saida