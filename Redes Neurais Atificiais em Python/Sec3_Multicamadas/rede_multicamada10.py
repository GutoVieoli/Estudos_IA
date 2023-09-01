# peso n+1 = (peso * momento) + (entrada * delta * taxa de aprendizagem)

import numpy as np

def sigmoid(soma):
    return 1 / ( 1 + np.exp(-soma))

def sigmoidDerivada(sig):
    return sig * (1 - sig)


entradas = np.array([[0, 0] , [0, 1], [1, 0], [1, 1]])

saidas = np.array([[0], [1], [1], [0]])

pesos0 = np.array([ [-0.424, -0.740, -0.961],  [0.358, -0.577, -0.469] ])

pesos1 = np.array([ [-0.017], [-0.893], [0.148] ])

epocas = 100000
momento = 1
taxaAprendizagem = 0.3

for j in range(epocas):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)
    #print(camadaOculta)

    somaSinapse1 = np.dot(camadaOculta, pesos1);
    camadaSaida = sigmoid(somaSinapse1)

    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print("Erro: " + str(mediaAbsoluta))

    deltaSaida = erroCamadaSaida * sigmoidDerivada(camadaSaida)
    #print(deltaSaida)
    #print("\n")

    pesos1 = pesos1.T
    deltaSaidaXPeso = deltaSaida.dot(pesos1)
    deltaCamadaOculta = sigmoidDerivada(camadaOculta) * deltaSaidaXPeso
    #print(deltaCamadaOculta)

    camadaOcultaTransposta = camadaOculta.T
    entradaXdelta = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1.T * momento) + (entradaXdelta * taxaAprendizagem)
    #print(pesos1)


    deltaCamadaOcultaTransposta = deltaCamadaOculta.T
    entradaXdelta2 = deltaCamadaOcultaTransposta.dot(entradas)
    # print(entradas)
    # print(deltaCamadaOcultaTransposta)
    #print(entradaXdelta2)
    pesos0 = (pesos0 * momento) + (entradaXdelta2.T * taxaAprendizagem)
    #print(pesos0)