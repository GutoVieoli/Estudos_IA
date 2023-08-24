import numpy as np

def sigmoid(soma):
    return 1 / ( 1 + np.exp(-soma))


entradas = np.array([[0, 0] , [0, 1], [1, 0], [1, 1]])

saidas = np.array([[0], [1], [1], [0]])

pesos0 = np.array([ [-0.424, -0.740, -0.961],  [0.358, -0.577, -0.469] ])

pesos1 = np.array([ [-0.017], [-0.893], [0.148] ])

epocas = 100

#for j in range(epocas):
camadaEntrada = entradas
somaSinapse0 = np.dot(camadaEntrada, pesos0)
camadaOculta = np.round( sigmoid(somaSinapse0), 3)

somaSinapse1 = np.dot(camadaOculta, pesos1);
camadaSaida = np.round( sigmoid(somaSinapse1), 3)

erroCamadaSaida = np.abs(saidas - camadaSaida)
mediaAbsoluta = np.round( np.mean(erroCamadaSaida), 3)
print(mediaAbsoluta)