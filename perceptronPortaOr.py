import numpy as np

def phi(v):
    if v >= 0:
        return 1
    return 0

MAX_ITER = 100
bias = 1
n = 0.1
W = np.array([0.5,0.5,0.5])
epocas = 0
erro = True

X = np.array([[0,0],[0,1],[1,0],[1,1]])
D = np.array([[0],[1],[1],[1]])

while erro and epocas < MAX_ITER:
    print(f"Época {epocas}")
    epocas += 1
    erro = False
    counterD = -1
    for line in X:
        counterD += 1
        counterX = 0
        v = 0
        while counterX < len(line):
            v += line[counterX]*W[counterX]
            counterX += 1
        v += bias * W[-1] 
        v = phi(v)
        if v != D[counterD]:
            W = W+n*(D[counterD]-v)*np.append(line, bias)
            print(f"Erro no exemplo {counterD}({X[counterD]})")
            print(f"Pesos recalculados: {W}")
            erro = True
    
