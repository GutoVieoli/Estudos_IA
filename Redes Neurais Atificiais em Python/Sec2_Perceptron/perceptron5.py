import numpy as np

entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saidas = np.array([0, 1, 1, 1])
pesos = np.array([0.0, 0.0])
taxaAprendizagem = 0.1

def stepFunction(soma):
    if( soma >= 1 ):
        return 1
    return 0

def calculaSaida(registro):
    s = registro.dot(pesos)
    return stepFunction(s)

def treinar():
    erroTotal = 1
    while( erroTotal != 0 ):
        erroTotal = 0
        for c in range(len(saidas)):
            saidaCalculada = calculaSaida(np.asarray(entradas[c]))
            #print(str(saidaCalculada))
            erro = abs(saidas[c] - saidaCalculada)
            erroTotal += erro
            for j in range( len(pesos) ):
                pesos[j] = pesos[j] + (taxaAprendizagem * entradas[c][j] * erro)
                pesos[j] = round(pesos[j], 2)
                print('Peso atualizado: ' + str(pesos[j]))
        print("Erro total: " + str(erroTotal) + " \n") 

treinar()