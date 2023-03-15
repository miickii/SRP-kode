import numpy as np

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_prime(Z):
    return Z > 0

def softmax(Z):
    # Beregn eksponentialerne for hver værdi i Z efter at have trukket den maksimale værdi
    # fra alle elementer i Z. Dette er en numerisk teknik, der forhindrer overflydskontrol 
    # ved at sikre, at eksponentialerne ikke bliver for store.
    exps = np.exp(Z - np.max(Z))

    # Beregn softmax-funktionen ved at dividere eksponentialerne med deres sum over den 
    # passende akse. Her summerer vi over søjlerne (axis=0) og bevarer den oprindelige 
    # dimensionalitet af matricen (keepdims=True), så der kan udføres broadcasting 
    # Summen af alle værdierne for en givende søjlevektor vil nu give 1 i den resulterende matrix
    return exps / np.sum(exps, axis=0, keepdims=True)

def softmax_prime(Z):
    return 1